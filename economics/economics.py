#!/usr/bin/env python3
"""Cache economics implied by DeepSeek's V4.1-Flash serving disclosure.

Every input is a number DeepSeek published (see SOURCES). The unknowns are the
token mix and the cluster size. This script finds the combinations of cache-hit
rate, output share and GPU count that are consistent with all the disclosed
numbers at once, and reports how revenue splits across token classes.

Run:  python3 -I economics/economics.py [--json results/economics.json]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

SOURCES = {
    "tokens_per_day": "DeepInfer doc/inference_deployment_qa.md: 'A single cluster currently serves 155 trillion tokens per day'",
    "revenue_per_gpu_hour": "Same doc: 'overall revenue of US$38 per GPU-hour (at valley pricing)'",
    "prices": "api-docs.deepseek.com/quick_start/pricing, deepseek-flash off-peak, read 2026-10-24: $0.003 / $0.15 / $0.60 per 1M (cache hit / miss / output)",
    "prefill_tps_gpu": "DeepInfer doc/images/prefill_pareto.png: EP4DP4, 131,072 context, 60K tok/s/GPU at 4,096-token batches rising to 89.5K at 65,536",
    "decode_tps_gpu": "Q&A speculative decoding table: 14,466 tok/s/rank at 60 seqs/rank with dynamic-length verification (the stated production operating point)",
    "online_efficiency": "Q&A: load balancing reaches 'roughly 80% of a perfectly balanced offline benchmark'",
    "pd_ratio": "Q&A: 'the GPU ratio between P and D is roughly 1:1'",
    "bytes_per_token": "DeepSeek-V4.1-Flash tech report (arXiv 2609.19969): 890 bytes per token of global KV; assumed equal to the persisted block KV",
    "store": "Q&A: '~100 PiB of SSD storage', TTL 3 days (agent) and 7 days (chatbot), 3FS 10+2 erasure coding",
    "storage_per_gpu": "Q&A: 'about 14.4 TB per GPU measured as logical size; the physical capacity required is larger than that'",
}

DAY_S = 86_400
PIB = 2**50


@dataclass
class Params:
    tokens_per_day: float = 155e12
    revenue_per_gpu_hour: float = 38.0
    # subhadipmitra@: off-peak prices on purpose. The $38/GPU-hour is quoted "at valley
    # pricing", so pairing it with peak prices (2x) would double-count revenue.
    price_hit: float = 0.003      # $ per 1M tokens, off-peak
    price_miss: float = 0.15
    price_out: float = 0.60
    # subhadipmitra@: both throughputs are read off DeepSeek's charts at 131K context. Shorter
    # contexts run faster, so for chat-sized prompts these understate capacity. 75K is a
    # midpoint between the 4K-batch (60K) and 64K-batch (89.5K) prefill points; pass
    # --prefill-tps to pin one end.
    prefill_tps_gpu: float = 75_000.0
    # subhadipmitra@: 14,466 is the 60-seqs/rank point the Q&A calls production. This one
    # number decides which "reading" you land in; the 128/rank point (22,800) is what makes
    # the 7,000-GPU reading possible. Check the sensitivity grid before quoting a GPU count.
    decode_tps_gpu: float = 14_466.0
    online_efficiency: float = 0.80
    # subhadipmitra@: not disclosed. Leaving it at 1.0 makes gpus_from_capacity a floor, and
    # utilization_needed is the number to look at instead.
    utilization: float = 1.0      # average over the day / benchmark peak; not disclosed
    pd_ratio: float = 1.0         # prefill GPUs per decode GPU
    # subhadipmitra@: assumes the persisted block KV is exactly the 890 B/token global KV.
    # The per-sequence SWA snapshots (~1.4 MB each) are left out; they are small next to the
    # block cache at these hit rates, but they are not zero.
    bytes_per_token: float = 890.0
    ec_overhead: float = 1.2      # 10+2 erasure coding
    store_pib: float = 100.0
    ttl_days: tuple = (3.0, 7.0)
    storage_per_gpu_tb: float = 14.4   # logical


@dataclass
class Point:
    output_share: float           # O / T
    hit_rate: float               # H / (H + M), share of input tokens that are cache hits
    hits: float
    misses: float
    outputs: float
    blended_price_per_m: float
    revenue_per_day: float
    gpus_from_revenue: float
    prefill_gpus: float
    decode_gpus: float
    gpus_from_capacity: float
    utilization_needed: float     # capacity GPUs at util=1 / revenue GPUs
    revenue_share: dict = field(default_factory=dict)


def mix(p: Params, hit_rate: float, output_share: float) -> tuple[float, float, float]:
    T = p.tokens_per_day
    O = output_share * T
    inputs = T - O
    H = hit_rate * inputs
    M = inputs - H
    return H, M, O


def hit_rate_for_pd_ratio(p: Params, output_share: float) -> float:
    """Hit rate at which prefill and decode GPU counts sit at the disclosed ratio.

    Prefill GPUs = M / (prefill_tps * k), decode GPUs = O / (decode_tps * k), same k.
    Setting prefill/decode = pd_ratio and solving for M gives the hit rate.
    """
    # subhadipmitra@: gotcha. Prefill GPUs are charged only for misses, so the 140-token
    # decoder replay that every hit also pays is not counted. At 95%+ hit rates that is a
    # real undercount of prefill work, and it pushes the solved hit rate up slightly.
    o = output_share
    misses_over_inputs = p.pd_ratio * o * p.prefill_tps_gpu / (p.decode_tps_gpu * (1.0 - o))
    return max(0.0, 1.0 - misses_over_inputs)


def evaluate(p: Params, hit_rate: float, output_share: float) -> Point:
    H, M, O = mix(p, hit_rate, output_share)
    rev = (p.price_hit * H + p.price_miss * M + p.price_out * O) / 1e6
    g_rev = rev / (p.revenue_per_gpu_hour * 24.0)
    k = DAY_S * p.online_efficiency * p.utilization
    g_pre = M / (p.prefill_tps_gpu * k)
    g_dec = O / (p.decode_tps_gpu * k)
    g_cap = g_pre + g_dec
    shares = {
        "hits": p.price_hit * H / 1e6 / rev,
        "misses": p.price_miss * M / 1e6 / rev,
        "outputs": p.price_out * O / 1e6 / rev,
    }
    return Point(
        output_share=output_share, hit_rate=hit_rate, hits=H, misses=M, outputs=O,
        blended_price_per_m=rev / (p.tokens_per_day / 1e6), revenue_per_day=rev,
        gpus_from_revenue=g_rev, prefill_gpus=g_pre, decode_gpus=g_dec,
        gpus_from_capacity=g_cap, utilization_needed=g_cap * p.utilization / g_rev,
        revenue_share=shares,
    )


def gpus_behind_store(p: Params) -> tuple[float, float]:
    """GPUs implied by ~100 PiB at 14.4 TB (logical) per GPU.

    Low end: the 100 PiB is physical, so logical capacity is 100 / 1.2.
    High end: the 100 PiB is already logical.
    """
    # subhadipmitra@: the Q&A does not say whether "~100 PiB" is raw SSD or logical capacity,
    # so both ends are returned. Do not collapse this to one number.
    per_gpu = p.storage_per_gpu_tb * 1e12
    return p.store_pib * PIB / p.ec_overhead / per_gpu, p.store_pib * PIB / per_gpu


def storage_check(p: Params, pt: Point) -> dict:
    cap_tokens = p.store_pib * PIB / (p.bytes_per_token * p.ec_overhead)
    # subhadipmitra@: outputs count as written because the Q&A stores SWA state at the end of
    # decode, which only makes sense if the decoded tokens' block KV is kept too. If you
    # believe outputs are not persisted, drop pt.outputs; the needed PiB falls a little.
    written_per_day = pt.misses + pt.outputs   # misses are computed and written; decode output is written too
    per_gpu_day_bytes = p.bytes_per_token * written_per_day / pt.gpus_from_revenue
    return {
        "retention_days_at_14p4tb_per_gpu": p.storage_per_gpu_tb * 1e12 / per_gpu_day_bytes,
        "capacity_tokens": cap_tokens,
        "capacity_as_days_of_served_tokens": cap_tokens / p.tokens_per_day,
        "written_tokens_per_day": written_per_day,
        "capacity_as_days_of_written_tokens": cap_tokens / written_per_day,
        "needed_pib_at_ttl": {f"{d:g}d": written_per_day * d * p.bytes_per_token * p.ec_overhead / PIB for d in p.ttl_days},
    }


def sweep(p: Params, output_shares: list[float]) -> list[Point]:
    return [evaluate(p, hit_rate_for_pd_ratio(p, o), o) for o in output_shares]


def fmt_table(p: Params, points: list[Point]) -> str:
    head = (f"{'out%':>6} {'hit%':>6} {'$/M blend':>10} {'G(rev)':>8} {'G(cap@1)':>9} {'util':>6} "
            f"{'rev hit%':>9} {'rev miss%':>10} {'rev out%':>9} {'PiB@3d':>7} {'PiB@7d':>7} {'days@14.4TB':>12}")
    rows = [head, "-" * len(head)]
    for q in points:
        sc = storage_check(p, q)
        st = sc["needed_pib_at_ttl"]
        rows.append(
            f"{100*q.output_share:6.2f} {100*q.hit_rate:6.2f} {q.blended_price_per_m:10.4f} "
            f"{q.gpus_from_revenue:8.0f} {q.gpus_from_capacity:9.0f} {q.utilization_needed:6.2f} "
            f"{100*q.revenue_share['hits']:9.1f} {100*q.revenue_share['misses']:10.1f} {100*q.revenue_share['outputs']:9.1f} "
            f"{st['3d']:7.0f} {st['7d']:7.0f} {sc['retention_days_at_14p4tb_per_gpu']:12.1f}"
        )
    return "\n".join(rows)


def february_2025() -> dict:
    """DeepSeek's own day of V3/R1 traffic, 2025-02-27 12:00 to 2025-02-28 12:00 UTC+8.

    Source: github.com/deepseek-ai/open-infra-index, 202502OpenSourceWeek,
    day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
    """
    inp, hit, out = 608e9, 342e9, 168e9
    miss, tot, gpus = inp - hit, inp + out, 226.75 * 8
    # subhadipmitra@: R1 prices for all traffic, V3 included, because that is how DeepSeek
    # computed its own "theoretical" revenue. It overstates real 2025 revenue (V3 was
    # cheaper, and there were discounts); use it for shares, not for absolute dollars.
    p_hit, p_miss, p_out = 0.14, 0.55, 2.19          # R1 list prices, $/M, the basis DeepSeek used
    rev = (p_hit * hit + p_miss * miss + p_out * out) / 1e6
    return {
        "hit_rate": hit / inp, "token_share": {"hits": hit / tot, "misses": miss / tot, "outputs": out / tot},
        "revenue_share": {"hits": p_hit * hit / 1e6 / rev, "misses": p_miss * miss / 1e6 / rev, "outputs": p_out * out / 1e6 / rev},
        "gpus": gpus, "revenue_per_day": rev, "revenue_per_gpu_hour": rev / gpus / 24, "tokens_per_gpu_day": tot / gpus,
        "price_ratio_hit_miss_out": (1.0, p_miss / p_hit, p_out / p_hit),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write all results here")
    ap.add_argument("--prefill-tps", type=float, default=None)
    ap.add_argument("--decode-tps", type=float, default=None)
    args = ap.parse_args()
    p = Params()
    if args.prefill_tps:
        p.prefill_tps_gpu = args.prefill_tps
    if args.decode_tps:
        p.decode_tps_gpu = args.decode_tps

    shares = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]
    print("Self-consistent operating points (hit rate pinned by the 1:1 prefill:decode ratio):")
    print(f"  prefill {p.prefill_tps_gpu:,.0f} tok/s/GPU, decode {p.decode_tps_gpu:,.0f} tok/s/GPU, online efficiency {p.online_efficiency}")
    base = sweep(p, shares)
    print(fmt_table(p, base))
    print("\n  G(rev): GPUs implied by $38/GPU-hour.  G(cap@1): GPUs needed at benchmark throughput all day.")
    print("  util: average/peak utilization that reconciles the two. Above 1.0 means the mix is infeasible at these throughputs.")
    print("  PiB@3d / PiB@7d: SSD needed to retain the computed tokens (misses + outputs) for that TTL at 890 B/token x 1.2 EC. DeepSeek runs ~100 PiB.")
    print("  days@14.4TB: days of each GPU's computed tokens that 14.4 TB (logical) per GPU holds.")

    print("\nSensitivity: decode operating point (from the decode Pareto chart) x prefill batch regime")
    grid = {}
    for dec_label, dec in (("32/rank 9.5K", 9_500), ("60/rank 14.5K", 14_466), ("128/rank 22.8K", 22_800), ("256/rank 26.2K", 26_200)):
        for pre_label, pre in (("4K batch 60K", 60_000), ("mid 75K", 75_000), ("64K batch 89.5K", 89_500)):
            q = Params(prefill_tps_gpu=pre, decode_tps_gpu=dec)
            pts = sweep(q, shares)
            feas = [x for x in pts if x.utilization_needed <= 1.0]
            best = max(feas, key=lambda x: x.output_share) if feas else None
            grid[f"{dec_label} | {pre_label}"] = [{**asdict(x), "storage": storage_check(q, x)} for x in pts]
            if best:
                print(f"  decode {dec_label:16s} prefill {pre_label:16s} -> max feasible output share {100*best.output_share:4.2f}%  hit {100*best.hit_rate:5.1f}%  GPUs {best.gpus_from_revenue:5.0f}")
            else:
                print(f"  decode {dec_label:16s} prefill {pre_label:16s} -> no feasible point in sweep")

    print("\nThe same table at the 128-requests-per-rank decode point (22,800 tok/s/GPU) and 64K-token prefill batches (89,500):")
    hi = Params(prefill_tps_gpu=89_500, decode_tps_gpu=22_800)
    print(fmt_table(hi, sweep(hi, shares)))

    lo, hi_g = gpus_behind_store(p)
    print(f"\nGPUs behind the store: 100 PiB / 14.4 TB per GPU = {lo:,.0f} (100 PiB physical) to {hi_g:,.0f} (100 PiB logical)")
    print("Where the revenue-implied GPU count meets that range, per operating point:")
    # subhadipmitra@: a fine sweep only for finding where the revenue line crosses the store
    # band; the coarse `shares` list above is what the tables print.
    fine = [x / 10000 for x in range(10, 801)]   # 0.10% .. 8.00% output share
    meet = {}
    for dec_label, dec in (("60/rank 14.5K", 14_466), ("64/rank 15.7K", 15_700), ("128/rank 22.8K", 22_800), ("256/rank 26.2K", 26_200)):
        for pre_label, pre in (("4K batch 60K", 60_000), ("16K batch 81.5K", 81_500), ("64K batch 89.5K", 89_500)):
            q = Params(prefill_tps_gpu=pre, decode_tps_gpu=dec)
            pts = [x for x in sweep(q, fine) if lo <= x.gpus_from_revenue <= hi_g]
            if not pts:
                continue
            a, b = pts[0], pts[-1]
            sa, sb = storage_check(q, a), storage_check(q, b)
            meet[f"{dec_label} | {pre_label}"] = {"from": {**asdict(a), "storage": sa}, "to": {**asdict(b), "storage": sb}}
            print(f"  decode {dec_label:15s} prefill {pre_label:16s} output {100*a.output_share:.2f}-{100*b.output_share:.2f}%  "
                  f"hit {100*b.hit_rate:.1f}-{100*a.hit_rate:.1f}%  util {a.utilization_needed:.2f}-{b.utilization_needed:.2f}  "
                  f"PiB@3d {sa['needed_pib_at_ttl']['3d']:.0f}-{sb['needed_pib_at_ttl']['3d']:.0f}  PiB@7d {sa['needed_pib_at_ttl']['7d']:.0f}-{sb['needed_pib_at_ttl']['7d']:.0f}  "
                  f"rev out {100*a.revenue_share['outputs']:.0f}-{100*b.revenue_share['outputs']:.0f}%  rev hit {100*b.revenue_share['hits']:.0f}-{100*a.revenue_share['hits']:.0f}%")

    f = february_2025()
    print("\nFebruary 2025, DeepSeek's own V3/R1 disclosure (for the then-and-now comparison):")
    print(f"  hit rate {100*f['hit_rate']:.1f}%   tokens hit/miss/out {100*f['token_share']['hits']:.1f}/{100*f['token_share']['misses']:.1f}/{100*f['token_share']['outputs']:.1f}%   "
          f"revenue {100*f['revenue_share']['hits']:.1f}/{100*f['revenue_share']['misses']:.1f}/{100*f['revenue_share']['outputs']:.1f}%")
    print(f"  {f['gpus']:,.0f} H800s, ${f['revenue_per_gpu_hour']:.2f} per GPU-hour at R1 list prices, {f['tokens_per_gpu_day']/1e6:.0f}M tokens per GPU a day")
    print(f"  price ratio hit:miss:out = 1 : {f['price_ratio_hit_miss_out'][1]:.1f} : {f['price_ratio_hit_miss_out'][2]:.1f}   (2026 off-peak: 1 : {Params().price_miss/Params().price_hit:.0f} : {Params().price_out/Params().price_hit:.0f})")
    for lbl, dec, o in (("reading A", 14_466, 0.005), ("reading B", 22_800, 0.035)):
        q = Params(prefill_tps_gpu=81_500, decode_tps_gpu=dec); x = evaluate(q, hit_rate_for_pd_ratio(q, o), o)
        print(f"  Oct 2026 {lbl}: hit rate {100*x.hit_rate:.1f}%  tokens hit/miss/out {100*x.hits/q.tokens_per_day:.1f}/{100*x.misses/q.tokens_per_day:.1f}/{100*x.outputs/q.tokens_per_day:.2f}%  "
              f"revenue {100*x.revenue_share['hits']:.1f}/{100*x.revenue_share['misses']:.1f}/{100*x.revenue_share['outputs']:.1f}%  "
              f"{x.gpus_from_revenue:,.0f} GPUs  {q.tokens_per_day/x.gpus_from_revenue/1e9:.1f}B tokens per GPU a day")

    ref = next(x for x in base if abs(x.output_share - 0.005) < 1e-9)
    st = storage_check(p, ref)
    print("\nStorage consistency at the 0.5% output-share point:")
    print(f"  100 PiB at {p.bytes_per_token:.0f} B/token x {p.ec_overhead} EC = {st['capacity_tokens']/1e12:,.0f}T tokens")
    print(f"  = {st['capacity_as_days_of_served_tokens']:.2f} days of tokens served, {st['capacity_as_days_of_written_tokens']:.1f} days of tokens computed")
    for k, v in st["needed_pib_at_ttl"].items():
        print(f"  PiB needed to hold {k} of computed tokens: {v:,.0f}")

    if args.json:
        out = {"params": asdict(p), "sources": SOURCES,
               "base": [{**asdict(x), "storage": storage_check(p, x)} for x in base],
               "grid": grid, "storage_at_0p5": st,
               "gpus_behind_store": {"physical_100pib": lo, "logical_100pib": hi_g}, "meet_store": meet,
               "february_2025": february_2025()}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
