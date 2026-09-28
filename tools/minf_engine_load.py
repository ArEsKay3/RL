#!/usr/bin/env python3
"""Summarize per-engine load from Megatron dynamic-engine step logs in a ray-driver.log.

Each engine (one TP group) prints a line every `logging_step_interval` steps:
  * rank R | step S | HH:MM:SS ... time: T ms [decode|non-decode ...] ... reqs: a A/MAX, p P, w W, f F, e E ...
  blocks: occupied O/TOTAL ... prefill (cumul): computed C, skipped K (X% skipped) ...

Usage:
  minf_engine_load.py RAY_DRIVER_LOG [--tp 4] [--bin-min 5] [--steady-frac 0.4] [--per-engine]
"""
import argparse
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta

LINE = re.compile(
    r"\* rank (?P<rank>\d+) \| step (?P<step>\d+) \| (?P<ts>\d\d:\d\d:\d\d) \.\.\. "
    r"time: (?P<ms>[\d.]+) ms \[(?P<kind>decode|non-decode)"
    r".*?reqs: a (?P<a>\d+)/(?P<amax>\d+), p (?P<p>\d+), w (?P<w>\d+), f (?P<f>\d+), e (?P<e>\d+)"
    r".*?blocks: occupied (?P<occ>\d+)/(?P<tot>\d+)"
    r"(?:.*?prefill \(cumul\): computed (?P<pc>\d+), skipped (?P<ps>\d+))?"
)


def parse(path, tp):
    samples = []
    seen = set()
    day = 0
    last = None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = LINE.search(line)
            if not m:
                continue
            engine = int(m["rank"]) // tp
            key = (engine, int(m["step"]))
            if key in seen:
                continue
            seen.add(key)
            t = datetime.strptime(m["ts"], "%H:%M:%S")
            if last is not None and t < last - timedelta(hours=1):
                day += 1
            last = t
            t = t + timedelta(days=day)
            samples.append(
                dict(
                    engine=engine,
                    step=int(m["step"]),
                    t=t,
                    ms=float(m["ms"]),
                    kind=m["kind"],
                    active=int(m["a"]),
                    amax=int(m["amax"]),
                    paused=int(m["p"]),
                    waiting=int(m["w"]),
                    finished=int(m["f"]),
                    errors=int(m["e"]),
                    occ=int(m["occ"]),
                    tot=int(m["tot"]),
                    pc=int(m["pc"]) if m["pc"] else None,
                    ps=int(m["ps"]) if m["ps"] else None,
                )
            )
    samples.sort(key=lambda s: s["t"])
    return samples


def fmt_min(td):
    return f"{td.total_seconds() / 60:5.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--bin-min", type=float, default=5.0)
    ap.add_argument("--steady-frac", type=float, default=0.4, help="last fraction of the trace treated as steady state")
    ap.add_argument("--per-engine", action="store_true")
    args = ap.parse_args()

    s = parse(args.log, args.tp)
    if not s:
        sys.exit("no engine step lines found")
    t0, t1 = s[0]["t"], s[-1]["t"]
    engines = sorted({x["engine"] for x in s})
    print(f"{len(s)} engine samples, {len(engines)} engines, {fmt_min(t1 - t0)} min of trace "
          f"({s[0]['t'].strftime('%H:%M:%S')} -> {s[-1]['t'].strftime('%H:%M:%S')}), "
          f"max active/engine = {s[0]['amax']}, KV blocks/engine = {s[0]['tot']}")

    # Timeline: per bin, aggregate over engine samples.
    bins = defaultdict(list)
    for x in s:
        b = int((x["t"] - t0).total_seconds() // (args.bin_min * 60))
        bins[b].append(x)
    print(f"\nTimeline ({args.bin_min:g}-min bins; active = requests being served per engine, w = waiting, "
          f"KV% = occupied blocks; step times are per-engine means):")
    print(f"{'t+min':>6} {'eng':>4} {'active mean':>11} {'min':>5} {'max':>5} {'w mean':>7} {'KV% mean':>9} "
          f"{'KV% max':>8} {'decode ms':>10} {'prefill ms':>11} {'%decode steps':>14}")
    for b in sorted(bins):
        xs = bins[b]
        per_engine = defaultdict(list)
        for x in xs:
            per_engine[x["engine"]].append(x["active"])
        eng_means = [statistics.mean(v) for v in per_engine.values()]
        kv = [100.0 * x["occ"] / x["tot"] for x in xs]
        dec = [x["ms"] for x in xs if x["kind"] == "decode"]
        pre = [x["ms"] for x in xs if x["kind"] != "decode"]
        print(f"{b * args.bin_min:6.0f} {len(per_engine):4d} {statistics.mean(eng_means):11.1f} "
              f"{min(eng_means):5.0f} {max(eng_means):5.0f} {statistics.mean(x['waiting'] for x in xs):7.2f} "
              f"{statistics.mean(kv):9.1f} {max(kv):8.1f} "
              f"{(statistics.mean(dec) if dec else float('nan')):10.1f} "
              f"{(statistics.mean(pre) if pre else float('nan')):11.1f} "
              f"{100.0 * len(dec) / len(xs):14.0f}")

    # Phases.
    span = (t1 - t0).total_seconds()
    steady_start = t0 + timedelta(seconds=span * (1.0 - args.steady_frac))
    peak = max(s, key=lambda x: x["active"])
    print(f"\nPeak active on one engine: {peak['active']} (engine {peak['engine']}, "
          f"t+{fmt_min(peak['t'] - t0).strip()} min, KV {100.0 * peak['occ'] / peak['tot']:.1f}%)")
    first_full = [x for x in s if x["active"] >= 1]
    if first_full:
        print(f"First engine serving a request: t+{fmt_min(first_full[0]['t'] - t0).strip()} min")

    steady = [x for x in s if x["t"] >= steady_start]
    if steady:
        per_engine = defaultdict(list)
        for x in steady:
            per_engine[x["engine"]].append(x)
        means = {e: statistics.mean(v["active"] for v in xs) for e, xs in per_engine.items()}
        kvm = {e: statistics.mean(100.0 * v["occ"] / v["tot"] for v in xs) for e, xs in per_engine.items()}
        vals = list(means.values())
        cv = statistics.pstdev(vals) / statistics.mean(vals) if statistics.mean(vals) else float("nan")
        print(f"\nSteady state = last {args.steady_frac:.0%} of the trace "
              f"(from t+{fmt_min(steady_start - t0).strip()} min), {len(per_engine)} engines:")
        print(f"  active per engine: mean {statistics.mean(vals):.1f}, min {min(vals):.1f}, max {max(vals):.1f}, "
              f"spread (CV) {cv:.2f}")
        print(f"  KV occupancy per engine: mean {statistics.mean(kvm.values()):.1f}%, max {max(kvm.values()):.1f}%")
        print(f"  waiting queue mean {statistics.mean(x['waiting'] for x in steady):.2f}, "
              f"paused mean {statistics.mean(x['paused'] for x in steady):.2f}, "
              f"errors max {max(x['errors'] for x in steady)}")
        last_by_engine = {}
        for x in s:
            last_by_engine[x["engine"]] = x
        pcs = [(x["pc"], x["ps"]) for x in last_by_engine.values() if x["pc"] is not None]
        if pcs:
            comp = sum(p for p, _ in pcs)
            skip = sum(q for _, q in pcs)
            if comp + skip:
                print(f"  prefix cache (cumulative, all engines): {100.0 * skip / (comp + skip):.1f}% of prefill "
                      f"tokens skipped")
        if args.per_engine:
            print(f"\n{'engine':>6} {'ranks':>9} {'samples':>7} {'steady active':>13} {'peak active':>11} "
                  f"{'steady KV%':>10}")
            for e in engines:
                xs = [x for x in s if x["engine"] == e]
                pk = max(x["active"] for x in xs)
                print(f"{e:6d} {e * args.tp:4d}-{e * args.tp + args.tp - 1:<4d} {len(xs):7d} "
                      f"{means.get(e, float('nan')):13.1f} {pk:11d} {kvm.get(e, float('nan')):10.1f}")


if __name__ == "__main__":
    main()
