#!/usr/bin/env python3
"""Create examples/yoco/yoco_real.py from DeepInfer's own yoco_equiv.py.

The upstream harness compares two implementations of the truncation (mask vs compact)
on random tokens. This copy keeps its engine setup, environment pinning and logprob
dump, and changes three things:
  1. prompts and golden continuations come from the JSON file named by REAL_PROMPTS
  2. sampling temperature is 1.0, so the dumped logprobs are the model's own
  3. gpu_memory_utilization (REAL_GPU_MEM_UTIL) and engram_parallel_size
     (REAL_ENGRAM_PARALLEL) can be set from the environment, for runs on fewer,
     larger GPUs than the harness expects, or on a node without RDMA
The exact baseline is the harness's own keep_tokens=0 ("off") mode.

Usage (from the DeepInfer repo root):  python3 patch_harness.py examples/yoco/yoco_equiv.py
"""
import sys
from pathlib import Path

src = Path(sys.argv[1]); s = src.read_text()

def sub(old: str, new: str) -> None:
    global s
    assert s.count(old) == 1, f"anchor not unique or missing: {old[:60]!r}"
    s = s.replace(old, new)

sub("def _prompt_lens(cfg: LogpConfig) -> list[int]:\n",
    "def _prompt_lens(cfg: LogpConfig) -> list[int]:\n"
    "    return [len(p) for p in _REAL()['prompts']]\n\n\ndef _prompt_lens_upstream(cfg: LogpConfig) -> list[int]:\n")
sub("def _build_prompts(cfg: LogpConfig, llm: object, lens: list[int]) -> tuple[list[list[int]], list[list[int]], list[dict] | None]:\n",
    "def _build_prompts(cfg: LogpConfig, llm: object, lens: list[int]) -> tuple[list[list[int]], list[list[int]], list[dict] | None]:\n"
    "    r = _REAL()\n"
    "    assert all(len(g) == cfg.output_len for g in r['golden']), 'pass --output_len equal to the golden length'\n"
    "    return r['prompts'], r['golden'], None\n\n\n"
    "def _build_prompts_upstream(cfg: LogpConfig, llm: object, lens: list[int]) -> tuple[list[list[int]], list[list[int]], list[dict] | None]:\n")
sub("            temperature=0.5,\n", "            temperature=1.0,\n")
sub("        gpu_memory_utilization=cfg.gpu_memory_utilization,\n",
    "        gpu_memory_utilization=cfg.gpu_memory_utilization,\n"
    "        engram_parallel_size=int(os.environ.get('REAL_ENGRAM_PARALLEL', '0')),\n")
sub("    gpu_memory_utilization: float = 0.85\n",
    "    gpu_memory_utilization: float = float(os.environ.get('REAL_GPU_MEM_UTIL', '0.85'))\n")
sub("@dataclass\nclass LogpConfig:\n",
    "def _REAL() -> dict:\n"
    "    import json as _json, os as _os\n"
    "    return _json.loads(Path(_os.environ['REAL_PROMPTS']).read_text())\n\n\n"
    "@dataclass\nclass LogpConfig:\n")
dst = src.with_name("yoco_real.py"); dst.write_text(s)
print("wrote", dst)
