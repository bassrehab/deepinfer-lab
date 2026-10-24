#!/usr/bin/env python3
"""Compare DeepInfer logprob dumps: exact decoder window (keep_tokens=0) vs bounded replay.

Inputs are the per-rank JSON files written by yoco_real.py logp, named keep{K}.rank{r}.json.
For every request the golden (teacher-forced) token is identical across runs, so the
golden-token logprob is an exact, like-for-like measurement. The top-k lists give an
approximate view of the rest of the distribution.

Reported per request and aggregated by prompt length, source and decode-step bucket:
  mean |dlogp|      mean absolute change in the golden token's logprob
  ppl ratio         perplexity of the golden continuation, replay / exact
  top-1 agree       share of steps whose argmax token is unchanged
  KL(top-k)         sum over the exact run's top-k of p (log p - log q); q missing from the
                    replay top-k is taken as the replay's k-th logprob, so this understates KL
  bitwise           share of steps whose golden logprob is exactly equal

Usage: python3 compare_replay.py DIR [--prompts prompts.json] [--json out.json]
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

BUCKETS = [(0, 16), (16, 128), (128, 256)]


def load(d: Path, k: int) -> list[dict]:
    files = sorted(d.glob(f"keep{k}.rank*.json"), key=lambda p: int(p.stem.split("rank")[-1]))
    return [json.loads(f.read_text()) for f in files]


# subhadipmitra@: every rank processes the same requests, so with DEEPINFER_BATCH_INVARIANT=1
# the ranks must agree bit for bit. If this ever prints "NOT identical", stop: something
# nondeterministic is in the path and the replay numbers below mean nothing.
def rank_invariance(dumps: list[dict]) -> str:
    base = [[s["logprob"] for s in r["steps"]] for r in dumps[0]["requests"]]
    same = all([[s["logprob"] for s in r["steps"]] for r in d["requests"]] == base for d in dumps[1:])
    return f"{len(dumps)} ranks, {'identical' if same else 'NOT identical'}"


def step_stats(sr: dict, sc: dict) -> dict:
    dl = sc["logprob"] - sr["logprob"]
    cand = dict(zip(sc["top_ids"], sc["top_logprobs"]))
    floor = min(sc["top_logprobs"]) if sc["top_logprobs"] else -50.0
    kl = 0.0
    for t, lp in zip(sr["top_ids"], sr["top_logprobs"]):
        if lp < -50:
            continue
        kl += math.exp(lp) * (lp - cand.get(t, floor))
    return {"dl": dl, "abs": abs(dl), "lr": sr["logprob"], "lc": sc["logprob"],
            # subhadipmitra@: exact float equality is intended for "bit". And keep in mind that
            # every metric here mixes the truncation with rounding from a different code
            # path; without a keep_tokens >= L control run they are upper bounds.
            "top1": sr["top_ids"][0] == sc["top_ids"][0], "kl": max(kl, 0.0), "bit": sr["logprob"] == sc["logprob"]}


def agg(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    return {
        "steps": n,
        "mean_abs_dlogp": sum(r["abs"] for r in rows) / n,
        "max_abs_dlogp": max(r["abs"] for r in rows),
        "mean_dlogp": sum(r["dl"] for r in rows) / n,
        "ppl_ratio": math.exp(-(sum(r["lc"] for r in rows) - sum(r["lr"] for r in rows)) / n),
        "top1_agree": sum(r["top1"] for r in rows) / n,
        "kl_topk": sum(r["kl"] for r in rows) / n,
        "bitwise": sum(r["bit"] for r in rows) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--prompts")
    ap.add_argument("--json")
    a = ap.parse_args()
    d = Path(a.dir)
    keeps = sorted({int(p.stem.split(".")[0][4:]) for p in d.glob("keep*.rank*.json")})
    assert 0 in keeps, "need the exact baseline keep0.rank*.json"
    meta = json.loads(Path(a.prompts).read_text())["meta"] if a.prompts else None
    ref = load(d, 0)
    print(f"exact baseline: {rank_invariance(ref)}")
    out = {}
    for k in [k for k in keeps if k]:
        cand = load(d, k)
        print(f"\n=== keep_tokens={k} vs exact  ({rank_invariance(cand)}) ===")
        by_len, by_src, by_bucket, allrows = defaultdict(list), defaultdict(list), defaultdict(list), []
        per_req = []
        for rr, rc in zip(ref[0]["requests"], cand[0]["requests"], strict=True):
            assert rr["golden"] == rc["golden"]
            rows = [step_stats(sr, sc) for sr, sc in zip(rr["steps"], rc["steps"], strict=True)]
            m = meta[rr["idx"]] if meta else {"source": "?", "len": rr["prompt_len"]}
            per_req.append({"source": m["source"], "len": m["len"], **agg(rows)})
            by_len[m["len"]] += rows; by_src[m["source"]] += rows; allrows += rows
            for i, r in enumerate(rows):
                for lo, hi in BUCKETS:
                    if lo <= i < hi:
                        by_bucket[f"{lo}-{hi - 1}"].append(r)
        hdr = f"{'':22s} {'steps':>6s} {'mean|dlp|':>10s} {'max|dlp|':>9s} {'ppl ratio':>10s} {'top1':>7s} {'KL(topk)':>10s} {'bitwise':>8s}"
        def line(lbl, s):
            return (f"{lbl:22s} {s['steps']:6d} {s['mean_abs_dlogp']:10.2e} {s['max_abs_dlogp']:9.3f} {s['ppl_ratio']:10.5f} "
                    f"{100*s['top1_agree']:6.2f}% {s['kl_topk']:10.2e} {100*s['bitwise']:7.1f}%")
        print(hdr)
        for r in per_req:
            print(line(f"{r['source']} L={r['len']}", r))
        print("-" * len(hdr))
        for L in sorted(by_len):
            print(line(f"all sources L={L}", agg(by_len[L])))
        for s in sorted(by_src):
            print(line(f"{s}, all lengths", agg(by_src[s])))
        for b in [f"{lo}-{hi - 1}" for lo, hi in BUCKETS]:
            print(line(f"decode steps {b}", agg(by_bucket[b])))
        print(line("everything", agg(allrows)))
        out[k] = {"per_request": per_req, "by_len": {L: agg(v) for L, v in by_len.items()},
                  "by_source": {s: agg(v) for s, v in by_src.items()},
                  "by_step": {b: agg(v) for b, v in by_bucket.items()}, "all": agg(allrows)}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
