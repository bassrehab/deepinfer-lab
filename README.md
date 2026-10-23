# deepinfer-lab

Working notes and runnable models around DeepSeek's October 2026 release of
[DeepInfer](https://github.com/deepseek-ai/DeepInfer) and
[deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe).
Companion code for [DeepInfer: the engine is open, the cache is the business](https://subhadipmitra.com/blog/2026/deepinfer-cache-is-the-business/) on subhadipmitra.com.

Nothing here needs a GPU. Every input number is one DeepSeek published, with the
source quoted next to it in the code.

| Directory | Question it answers |
|---|---|
| `economics/` | What token mix, cache-hit rate and cluster size are consistent with "155 trillion tokens a day" and "$38 per GPU-hour" at the published prices and the published throughput curves? |
| `cache_policy/` | What does the KV-cache hit policy described in DeepInfer's deployment Q&A cost, per turn, compared with an ideal prefix cache? |
| `recipe_contract/` | Does a control-token spelling inside a tool result become a real control token, depending on how you tokenize deepseek-recipe's rendered prompt? |

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
script solves for the hit rate that satisfies the ratio, then computes the GPU
count implied by revenue and the GPU count needed to compute the misses and
outputs at a given operating point on DeepInfer's own Pareto charts. The two
counts agree only in a narrow band. A storage check then asks how many PiB it
takes to hold the computed tokens for the published 3-day and 7-day TTLs.

At the production decode point DeepSeek states in the Q&A (about 60 sequences
per rank, 14.5K output tokens per second per GPU) the numbers close only if
output is about 0.5% of tokens and 97% of input tokens are cache hits, on a
cluster of roughly 1,700 GPUs. At the high-concurrency end of the chart (128 per
rank, 22.8K) they close at 1.5 to 2% output, 92 to 94% hits, 3,500 to 4,500 GPUs,
and that is the only region where seven days of computed tokens also fills the
~100 PiB SSD cache DeepSeek says it runs. In every feasible case cache hits are
most of the tokens and a minority of the revenue.

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

Headline, seed 0, 60 sessions per workload:

| workload | ideal prefix hit | policy hit | compute vs no cache: ideal prefix | exact decoder replay | DeepInfer policy |
|---|---|---|---|---|---|
| agent | 95.8% | 95.7% | 4.2% | 4.7% | 2.3% |
| chat | 95.9% | 90.8% | 4.1% | 41.1% | 7.4% |
| fresh document | 22.0% | 21.9% | 78.0% | 40.3% | 39.1% |

On long agent sessions the bounded replay makes prefill about 45% cheaper than
an exact prefix cache with full-stack prefill. On short chat turns the fixed
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

## Sources

- DeepInfer repository and `doc/inference_deployment_qa.md`, deepseek-ai, 2026-10-20
- deepseek-recipe 0.2.0, deepseek-ai, 2026-10-20
- DeepSeek API pricing page, read 2026-10-24
- DeepSeek-V4.1-Flash technical report, arXiv 2609.19969
- DualPath, arXiv 2602.21548 (the decode load balancer the Q&A points to)

## License

MIT.
