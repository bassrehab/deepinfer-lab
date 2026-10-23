#!/usr/bin/env python3
"""Replay the KV-cache hit policy from DeepInfer's deployment Q&A on synthetic traffic.

The policy is taken literally from doc/inference_deployment_qa.md ("KV cache hits and
recomputation" and "SWA KV cache"):

  * block KV cache is stored at 128-token granularity for every computed token
  * SWA KV cache is stored only at multiples of 128: once after prefill (reserve one
    token, round down) and once at the end of decode (last completed 128 boundary)
  * for an input of length L the hit length is capped at L - 134 (128 window + 6
    speculative backtracking tokens); the block cache gives L1 (a multiple of 128);
    if an SWA store exists in [L1 - 128, L1] the hit is L1, otherwise L1 - 128
  * encoder prefill runs over L - hit tokens; the decoder only replays the last
    140 tokens (bounded replay), never the whole suffix

Compute is counted in layer-tokens: tokens pushed through one transformer layer.
V4.1-Flash has 20 encoder and 20 decoder layers, so a full-stack prefill of n tokens
costs 40n. Four accounting schemes are compared on the same traces:

  no_cache        40 * L
  ideal_prefix    40 * (L - LCP)                exact prefix cache, full-stack prefill
  exact_yoco      20 * (L - LCP) + 20 * min(L, 2560)   encoder prefix cache + exact decoder SWA (2,560-token replay)
  deepinfer       20 * (L - hit) + 20 * min(L, 140)    the Q&A policy

Run:  python3 -I cache_policy/cache_policy_sim.py [--json results/cache_policy.json] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field

import numpy as np

BLOCK = 128
WINDOW = 128
BACKTRACK = 6                 # DEEPINFER_USE_NEXTN_DECODE (5) + 1
CAP = WINDOW + BACKTRACK      # 134
KEEP = 140                    # DEEPINFER_YOCO_PREFILL_KEEP_TOKENS default
EXACT_DECODER_REPLAY = 2_560  # 20 decoder layers x 128 window
ENC, DEC = 20, 20


@dataclass
class Stored:
    tokens: np.ndarray
    swa_positions: set[int] = field(default_factory=set)


def _lcp(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    neq = np.flatnonzero(a[:n] != b[:n])
    return int(neq[0]) if len(neq) else n


class Store:
    """Everything a cluster could hit: block cache on 3FS plus the SWA directory.

    Tokens are random integers, so a request can only hit entries written by its own
    session or the shared system prompt. The store is therefore scoped per session,
    which is equivalent and keeps the longest-common-prefix search cheap. "Any engine
    can hit any prefix" (Q&A) is modelled by never evicting within a session.
    """

    def __init__(self, swa_available: float, rng: random.Random):
        self.entries: list[Stored] = []
        self.swa_available = swa_available
        self.rng = rng

    def reset_session(self, keep: list["Stored"]) -> None:
        self.entries = list(keep)

    def lcp(self, prompt: np.ndarray) -> tuple[int, Stored | None]:
        best, best_e = 0, None
        for e in self.entries:
            i = _lcp(e.tokens, prompt)
            if i > best:
                best, best_e = i, e
        return best, best_e

    def hit(self, prompt: np.ndarray) -> dict:
        L = len(prompt)
        lcp, e = self.lcp(prompt)
        cap = L - CAP
        if cap <= 0 or e is None:
            return {"L": L, "lcp": lcp, "L1": 0, "hit": 0, "swa_backoff": False}
        L1 = (min(cap, lcp) // BLOCK) * BLOCK
        if L1 <= 0:
            return {"L": L, "lcp": lcp, "L1": 0, "hit": 0, "swa_backoff": False}
        swa_ok = any(L1 - WINDOW <= pos <= L1 for pos in e.swa_positions) and self.rng.random() < self.swa_available
        hit = L1 if swa_ok else max(0, L1 - BLOCK)
        return {"L": L, "lcp": lcp, "L1": L1, "hit": hit, "swa_backoff": not swa_ok}

    def put(self, prompt: np.ndarray, output: np.ndarray) -> None:
        L = len(prompt)
        swa = set()
        p = ((L - 1) // BLOCK) * BLOCK
        if p >= BLOCK:
            swa.add(p)
        d = ((L + len(output)) // BLOCK) * BLOCK
        if d > p:
            swa.add(d)
        self.entries.append(Stored(np.concatenate([prompt, output]), swa))


def cost(L: int, lcp: int, hit: int) -> dict:
    return {
        "no_cache": 40 * L,
        "ideal_prefix": 40 * (L - lcp),
        "exact_yoco": ENC * (L - lcp) + DEC * min(L, EXACT_DECODER_REPLAY),
        "deepinfer": ENC * (L - hit) + DEC * min(L, KEEP),
    }


def fresh(rng: random.Random, n: int) -> np.ndarray:
    return np.array([rng.randrange(1 << 30) for _ in range(n)], dtype=np.int64)


def cat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def run_session(kind: str, store: Store, rng: random.Random, system: np.ndarray) -> list[dict]:
    rows = []
    if kind == "agent":
        turns, user_rng, out_rng = 40, (200, 4000), (50, 1500)
    elif kind == "chat":
        turns, user_rng, out_rng = 8, (10, 200), (100, 800)
    else:  # document QA: one big fresh prompt, sometimes re-asked
        turns, user_rng, out_rng = 1, (20_000, 200_000), (200, 2000)
    ctx = system.copy()
    for t in range(turns):
        if kind == "agent" and t > 0 and rng.random() < 0.03:
            # context compaction: keep a prefix, replace the rest with a fresh summary
            keep = int(len(ctx) * 0.4)
            ctx = cat(ctx[:keep], fresh(rng, 500))
        prompt = cat(ctx, fresh(rng, rng.randint(*user_rng)))
        h = store.hit(prompt)
        out = fresh(rng, rng.randint(*out_rng))
        store.put(prompt, out)
        rows.append({"kind": kind, "turn": t, **h, **cost(h["L"], h["lcp"], h["hit"]), "new_tokens": len(prompt) - len(ctx)})
        ctx = cat(prompt, out)
        if kind == "agent" and rng.random() < 0.05:
            # a sibling request off the same prefix (retry / parallel branch)
            sib = cat(prompt, fresh(rng, rng.randint(20, 400)))
            h2 = store.hit(sib)
            out2 = fresh(rng, rng.randint(*out_rng))
            store.put(sib, out2)
            rows.append({"kind": kind, "turn": t, **h2, **cost(h2["L"], h2["lcp"], h2["hit"]), "new_tokens": len(sib) - len(prompt)})
    if kind == "doc" and rng.random() < 0.3:
        # the same document, asked a second question
        prompt2 = cat(prompt, fresh(rng, rng.randint(20, 300)))
        h3 = store.hit(prompt2)
        out3 = fresh(rng, rng.randint(*out_rng))
        store.put(prompt2, out3)
        rows.append({"kind": kind, "turn": 1, **h3, **cost(h3["L"], h3["lcp"], h3["hit"]), "new_tokens": len(prompt2) - len(prompt)})
    return rows


def summarize(rows: list[dict]) -> dict:
    L = sum(r["L"] for r in rows)
    lcp = sum(r["lcp"] for r in rows)
    hit = sum(r["hit"] for r in rows)
    tot = {k: sum(r[k] for r in rows) for k in ("no_cache", "ideal_prefix", "exact_yoco", "deepinfer")}
    backoffs = sum(1 for r in rows if r["swa_backoff"] and r["L1"] > 0)
    overhead = [r["L"] - r["hit"] - r["new_tokens"] for r in rows if r["lcp"] >= r["L"] - r["new_tokens"] and r["L1"] > 0]
    return {
        "requests": len(rows),
        "input_tokens": L,
        "ideal_hit_rate": lcp / L,
        "policy_hit_rate": hit / L,
        "swa_backoffs": backoffs,
        "recomputed_beyond_new_tokens_mean": (sum(overhead) / len(overhead)) if overhead else None,
        "layer_tokens": tot,
        "relative_to_no_cache": {k: v / tot["no_cache"] for k, v in tot.items()},
        "deepinfer_vs_ideal_prefix": tot["deepinfer"] / tot["ideal_prefix"] if tot["ideal_prefix"] else None,
        "deepinfer_vs_exact_yoco": tot["deepinfer"] / tot["exact_yoco"] if tot["exact_yoco"] else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sessions", type=int, default=60)
    ap.add_argument("--swa-available", type=float, nargs="+", default=[1.0, 0.9])
    args = ap.parse_args()

    results = {}
    for swa_p in args.swa_available:
        for kind in ("agent", "chat", "doc"):
            rng = random.Random(args.seed)
            store = Store(swa_p, rng)
            system = fresh(rng, 2000 if kind == "agent" else 500 if kind == "chat" else 300)
            store.put(system, np.empty(0, dtype=np.int64))   # the shared system prompt is always warm
            warm = list(store.entries)
            rows = []
            for _ in range(args.sessions):
                store.reset_session(warm)
                rows += run_session(kind, store, rng, system)
            s = summarize(rows)
            results[f"{kind}@swa{swa_p}"] = s
            r = s["relative_to_no_cache"]
            print(f"{kind:5s} swa_avail={swa_p:<4}  req={s['requests']:4d}  input={s['input_tokens']/1e6:6.1f}M  "
                  f"ideal hit {100*s['ideal_hit_rate']:5.1f}%  policy hit {100*s['policy_hit_rate']:5.1f}%  "
                  f"compute vs no-cache: ideal {100*r['ideal_prefix']:5.1f}%  exact-yoco {100*r['exact_yoco']:5.1f}%  deepinfer {100*r['deepinfer']:5.1f}%  "
                  f"| deepinfer/ideal {s['deepinfer_vs_ideal_prefix']:.2f}  deepinfer/exact-yoco {s['deepinfer_vs_exact_yoco']:.2f}  "
                  f"| extra recompute/turn {s['recomputed_beyond_new_tokens_mean'] and round(s['recomputed_beyond_new_tokens_mean'])}  backoffs {s['swa_backoffs']}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"seed": args.seed, "sessions": args.sessions, "results": results}, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
