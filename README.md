# deepinfer-lab

Working notes and runnable models around DeepSeek's October 2026 release of
[DeepInfer](https://github.com/deepseek-ai/DeepInfer) and
[deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe).
Companion code for [DeepInfer: the engine is open, the cache is the business](https://subhadipmitra.com/blog/2026/deepinfer-cache-is-the-business/) on subhadipmitra.com.

Only `replay_accuracy/` needs a GPU (two B300s were enough). Every input number is one DeepSeek published, with the
source quoted next to it in the code.

| Directory | Question it answers |
|---|---|
| `economics/` | What token mix, cache-hit rate and cluster size are consistent with "155 trillion tokens a day" and "$38 per GPU-hour" at the published prices and the published throughput curves? |
| `cache_policy/` | What does the KV-cache hit policy described in DeepInfer's deployment Q&A cost, per turn, compared with an ideal prefix cache? |
| `recipe_contract/` | Does a control-token spelling inside a tool result become a real control token, depending on how you tokenize deepseek-recipe's rendered prompt? |
| `replay_accuracy/` | What does SWA bounded replay, the decoder approximation DeepInfer ships on by default, do to the model's next-token distribution on real text? Needs Blackwell GPUs. |

## Setup

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## economics

```sh
python3 economics/economics.py --json results/economics.json
python3 economics/plot_economics.py results/economics.json results/feasibility_band.png
```

The model has two unknowns, the share of tokens that are model output and the
share of input tokens that are cache hits, and one disclosed ratio that ties them
together: prefill and decode GPUs are "roughly 1:1". For each output share the
script solves for the hit rate that satisfies the ratio, then computes three GPU
counts:

- the count implied by revenue at $38 per GPU-hour;
- the count needed to compute the misses and outputs at a chosen operating point
  on DeepInfer's own Pareto charts, at 80% online efficiency;
- the count behind the cache, from about 100 PiB of SSD at about 14.4 TB of
  logical cache per GPU, which is 6,516 GPUs if the 100 PiB is physical and
  7,819 if it is logical.

The disclosures leave two readings.

| | decode point | output share | input hit rate | GPUs | needs |
|---|---|---|---|---|---|
| A | 60 req/rank, the Q&A's production point | 0.25 to 0.75% | 95 to 99% | 1,000 to 2,400 | a 100 PiB store that serves more than this cluster |
| B | 128+ req/rank | 3 to 4% | 85 to 90% | 6,500 to 7,800 | decode at twice the stated concurrency, and mostly 3-day TTLs |

The 14.4 TB per GPU ratio holds four to six days of each GPU's computed tokens
at every point, so it does not discriminate between the readings. In both,
cache hits are the large majority of tokens and at most half the revenue.

The script also reproduces DeepSeek's February 2025 V3/R1 disclosure for
comparison: a 56.3% hit rate, output at 21.6% of tokens and 65.5% of revenue at
R1 list prices, and about 430M tokens per H800 per day.

## cache_policy

```sh
python3 cache_policy/cache_policy_sim.py --json results/cache_policy.json
```

A discrete simulation of the hit rules in the Q&A: 128-token blocks, SWA state
stored only at 128-token boundaries, a hard cap of L - 134 on the hit length, a
128-token back-off when the SWA state is missing, and a 140-token decoder
replay instead of exact reconstruction. Compute is counted in layer-tokens and
compared across four schemes on identical synthetic traces (agent sessions,
chat sessions, fresh long documents).

Headline, seed 0, 60 sessions per workload. The last three columns are prefill
compute as a share of running with no cache. "Ordinary model" is a 40-layer
decoder-only model that runs the uncached suffix through every layer. "Exact
rebuild" is this model with the decoder window reconstructed over 2,560 tokens.

| workload | hit rate, perfect cache | hit rate, DeepInfer rules | ordinary model, perfect cache | this model, exact rebuild | DeepInfer rules |
|---|---|---|---|---|---|
| agent | 95.8% | 95.7% | 4.2% | 4.7% | 2.3% |
| chat | 95.9% | 90.8% | 4.1% | 41.1% | 7.4% |
| fresh document | 22.0% | 21.9% | 78.0% | 40.3% | 39.1% |

On long agent sessions the bounded replay costs about half of either baseline. On short chat turns the fixed
per-turn overhead (the 134-token cap, block rounding, the 140-token replay)
makes it about 1.8x more expensive than ideal, though both are tiny in absolute
terms. On fresh long documents it halves prefill, which is the "nearly halving"
claim in the Q&A.

## recipe_contract

```sh
pip install deepseek-recipe==0.2.0 tokenizers
git clone --depth 1 https://github.com/deepseek-ai/deepseek-recipe /tmp/recipe
python3 recipe_contract/control_token_probe.py /tmp/recipe/static/tokenizers/v41/tokenizer.json
```

deepseek-recipe renders a prompt string plus `special_ranges`, the byte ranges
that may tokenize as control tokens. The supported `encode` path keeps a tool
result containing `<｜User｜>` as plain text. Tokenizing the same rendered string
with the Hugging Face `tokenizers` crate, or with the recipe's own `Tokenizer`
class, turns the injected spelling into a real control token. The checkpoint on
Hugging Face ships no chat template, and its reference `encoding.py` returns a
bare string, so this is the default failure mode for anyone serving V4.1-Flash
outside DeepInfer or vLLM's dedicated tokenizer mode.

```
path             tokens   EOS  User  Asst  verdict
recipe_encode       341     1     2     2  control tokens preserved
recipe_tokenize     324     2     3     3  INJECTED control tokens
hf_tokenizers       324     2     3     3  INJECTED control tokens
expected                    1     2     2
```

## replay_accuracy

In prefill, DeepInfer runs only the last 140 tokens through the 20 decoder layers
(`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`), whether the prefix came from the cache or not.
Rebuilding the decoder's sliding-window state exactly would need 2,560. This directory
measures what that shortcut does to the model's next-token distribution, with DeepInfer's
own equivalence harness on the real checkpoint.

1. `make_prompts.py` builds 12 requests at 256, 1,024, 4,096 and 16,384 tokens from three
   sources (Python, Rust, prose), each with the 256 tokens that follow it in the source as
   a golden continuation. Every added-token spelling in the source text is broken with a
   space, so no request carries a control token.
2. `patch_harness.py` copies `examples/yoco/yoco_equiv.py` to `yoco_real.py`, swaps in
   those requests, sets temperature to 1.0 so the logprobs are the model's own, and reads
   `REAL_GPU_MEM_UTIL` and `REAL_ENGRAM_PARALLEL` from the environment.
3. `patch_single_node.py` lets the engine run on one cloud node (see below).
4. `pod_setup.sh` prints the hardware, starts a watchdog that deletes the pod after
   `MAX_HOURS`, downloads the checkpoint and builds DeepInfer at 9bd9cf2.
5. `preflight.sh` runs DeepEP's Engram test, and its EP test with Gin off.
6. `run_replay.sh` runs the harness at keep_tokens 0 (the harness's exact baseline) and 140
   (the shipped default), and 512 if you have the time.
7. `compare_replay.py` reports mean and max |dlogp| on the golden token, the perplexity
   ratio, top-1 agreement, KL over the top 20, and the share of steps that match bit for
   bit, by length, source and decode step.

```sh
python3 replay_accuracy/make_prompts.py --tokenizer <tokenizer.json> \
    --deepinfer <DeepInfer checkout> --recipe <deepseek-recipe checkout> \
    --prose <some .md files> --out prompts.json
# on the pod, with the replay_accuracy scripts and prompts.json in /workspace
MAX_HOURS=5 bash pod_setup.sh
bash preflight.sh
cd DeepInfer && python3 /workspace/patch_harness.py examples/yoco/yoco_equiv.py
KEEPS="0 140" bash /workspace/run_replay.sh
# back here
python3 replay_accuracy/compare_replay.py results/replay --prompts results/replay/prompts.json \
    --json results/replay_accuracy.json
```

### Result

Two B300s, one seed, 12 requests, 3,072 teacher-forced positions. Dumps are in
`results/replay/` (rank 0; the two ranks were identical), the summary in
`results/replay_accuracy.json`.

| | 140-token replay against exact |
|---|---|
| Perplexity on the golden continuation | 0.36% higher (bootstrap over requests, 95%: 0.0 to 0.8%) |
| Top-1 token changes | 1.1% of positions, about the same in steps 0-15, 16-127 and 128-255 |
| abs(dlogp) of the golden token > 0.1 | 10.5% of positions (> 0.5: 0.65%) |
| Bit-identical to exact | 6.1% of positions |
| By prompt length (256 / 1K / 4K / 16K) | ppl ratio 1.003 / 1.007 / 1.000 / 1.004 |

Per-request perplexity ratios run from 0.994 to 1.024. There is no control run separating
the truncation from the change of code path; keep_tokens=512 on the 256-token prompts
would be that control, and it did not fit in the time I gave the pod.

### Running it on one cloud node

The checkpoint is 475 GiB: 276 GiB of routed experts, 189 GiB in two Engram tables and
their scales, about 11 GiB of everything else. Getting it to run on a rented node took
more than memory arithmetic:

- DeepEP fetches Engram rows with NCCL Gin (GDAKI), which needs an InfiniBand device even
  when every rank is on one node. `EngramBuffer` construction fails with "NCCL GIN is
  unavailable", and with `EP_DISABLE_GIN=1` the fetch kernel fails to load instead. MoE
  dispatch is fine with Gin off, in hybrid mode too. `patch_single_node.py` replaces only
  the Engram fetch: with `engram_parallel_size=1` each rank holds the whole FP8 table in
  host RAM (183 GiB per rank) and gathers rows on the CPU, scale factors on the GPU, in
  the layouts DeepEP returns. That needs `DEEPINFER_ENFORCE_EAGER=1` and a pause of
  torch's sync debug mode around the gather. The model is sane under it: perplexity on
  the golden continuations is 1.1 to 4.7 and top-1 matches the golden token 61 to 98% of
  the time.
- The loader reads experts and Engram tables single-threaded and has a 30-minute
  watchdog. From a network volume that is not enough; the patch raises it to three hours
  and `run_replay.sh` prefetches the shards with 16 readers first. One load then takes
  about 17 minutes.
- Under CUDA 13.2, one FlashMLA SM100 decode kernel spills registers and FlashMLA's build
  treats that as an error (`FLASH_MLA_SKIP_REG_SPILL_CHECK=1`). The base image also needed
  `pciutils`, `libdw-dev` and friends, `PIP_BREAK_SYSTEM_PACKAGES=1`, and `VIRTUAL_ENV`
  set, which DeepInfer's `setup.py` checks for and never uses.

The run used two B300s for 5.5 hours, most of it finding the above; a clean run of two
passes is about 1.5 hours.

## Sources

- DeepInfer repository and `doc/inference_deployment_qa.md`, deepseek-ai, 2026-10-20
- deepseek-recipe 0.2.0, deepseek-ai, 2026-10-20
- DeepSeek API pricing page, read 2026-10-24
- DeepSeek-V4.1-Flash technical report, arXiv 2609.19969
- DualPath, arXiv 2602.21548 (the decode load balancer the Q&A points to)

## License

MIT.
