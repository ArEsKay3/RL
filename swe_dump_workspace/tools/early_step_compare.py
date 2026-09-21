"""Per-step, per-prompt comparison of the advantage structure in the first train steps.

vLLM side: run A token rows (true advantages) joined to rollout summaries (instance ids).
MINF side: main chain per-instance Gym results (resolved + completion tokens), mapped to
steps through run A's in_order prompt sets. Writes CSVs + a markdown report.
"""
from __future__ import annotations

import csv, glob, json, os, re, statistics as st, sys
from collections import defaultdict

W = "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump"
OUT = f"{W}/analysis/early_steps"
GYM = ("/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_minf/"
       "nemo_rl/3rdparty/Gym-workspace/Gym/results/swebench_results_1789570827574_f7cb6b6f")
MAX_STEP = 8
DIR_RE = re.compile(r"^(.*)_(\d{13})_([0-9a-f]{8})$")


def load_runA():
    summ = {}
    for f in glob.glob(f"{W}/analysis/rollouts/*vllm_dump*__target_step_*.summary.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if r.get("target_step") is None or r["target_step"] + 1 > MAX_STEP:
                continue
            summ[r["sample_id"]] = r
    rows = []
    for f in glob.glob(f"{W}/analysis/tokens/*vllm_dump*__step_*.rows.csv"):
        step = int(f.split("__step_")[1].split("_")[0])
        if step > MAX_STEP:
            continue
        for r in csv.DictReader(open(f)):
            s = summ.get(r["sample_id"])
            rows.append(dict(step=step, sample_id=r["sample_id"], group_id=r["group_id"], adv=float(r["adv"]),
                             L=int(r["n_gen_tokens"]), reward=float(r["reward"]),
                             smb=float(r["sample_mask_before"]), sma=float(r["sample_mask_after"]),
                             trunc=(r["rollout_truncated"] in ("True", "1", "1.0")),
                             instance=s["instance_id"] if s else None, resolved=s.get("resolved") if s else None,
                             tstep=(s["target_step"] + 1) if s else None, turns=s.get("turns") if s else None,
                             err=(s.get("agent_error_kind") or ("truncated" if s.get("truncated") else "none")) if s else None))
    return rows


def fit_formula(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["step"], r["group_id"])].append(r)
    errs = []
    for g in groups.values():
        for x, a in zip(g, adv_from_rewards([y["reward"] for y in g])):
            errs.append(abs(a - x["adv"]))
    return {(1, False): (max(errs), st.mean(errs), len(errs))}


def adv_from_rewards(rs, ddof=None):
    out = []
    for i, r in enumerate(rs):
        others = rs[:i] + rs[i + 1:]
        if not others:
            out.append(0.0); continue
        b = st.mean(others); sd = st.stdev(others) if len(others) > 1 else 0.0
        out.append((r - b) / sd if sd > 0 else (r - b))
    return out


def load_minf(instance_to_step):
    per = defaultdict(list)
    n_dirs = n_ok = 0
    for d in os.scandir(GYM):
        if not d.is_dir():
            continue
        n_dirs += 1
        m = DIR_RE.match(d.name)
        if not m:
            continue
        inst = m.group(1)
        if inst not in instance_to_step:
            continue
        p = os.path.join(d.path, "nemo_gym_metrics.json")
        try:
            j = json.load(open(p))
        except Exception:
            continue
        usage = ((j.get("per_turn_metrics") or {}).get("accumulated_token_usage") or {})
        L = usage.get("completion_tokens")
        if L is None:
            continue
        n_ok += 1
        per[inst].append(dict(instance=inst, step=instance_to_step[inst], reward=1.0 if j.get("resolved") else 0.0,
                              L=int(L), ts=j.get("generation_start_timestamp"), run_time=j.get("openhands_run_time"),
                              turns=len((j.get("per_turn_metrics") or {}).get("token_usages") or []),
                              err=j.get("agent_error_kind") or "none"))
    return per, n_dirs, n_ok


def group_stats(samples, adv_key="adv"):
    L = [x["L"] for x in samples]; A = [x[adv_key] for x in samples]; R = [x["reward"] for x in samples]
    k = sum(R); n = len(R)
    succ = [x["L"] for x in samples if x["reward"] > 0]; fail = [x["L"] for x in samples if x["reward"] <= 0]
    return dict(n=n, k=int(k), pass_rate=k / n, mixed=(0 < k < n), meanL=st.mean(L),
                L_succ=st.mean(succ) if succ else float("nan"), L_fail=st.mean(fail) if fail else float("nan"),
                sumAL=sum(a * l for a, l in zip(A, L)), sumL=sum(L))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load_runA()
    print(f"run A rows {len(rows)}, with instance {sum(1 for r in rows if r['instance'])}")
    print("mask_before==0:", sum(1 for r in rows if r["smb"] == 0), " mask_after==0:", sum(1 for r in rows if r["sma"] == 0),
          " truncated:", sum(1 for r in rows if r["trunc"]))
    print("reward==resolved mismatches:", sum(1 for r in rows if r["resolved"] is not None and float(bool(r["resolved"])) != r["reward"]))
    variants = fit_formula(rows)
    for k, v in sorted(variants.items(), key=lambda kv: kv[1][0]):
        print(f"formula ddof={k[0]} masked_only={k[1]}: max_err {v[0]:.4f} mean_err {v[1]:.5f} n {v[2]}")
    ddof = sorted(variants.items(), key=lambda kv: kv[1][0])[0][0][0]

    inst_step = {}
    for r in rows:
        if r["instance"]:
            inst_step[r["instance"]] = r["tstep"]
    print("instances mapped:", len(inst_step))
    minf, n_dirs, n_ok = load_minf(inst_step)
    print(f"MINF session dirs {n_dirs}, matched rollouts {n_ok}, instances {len(minf)}")
    cnt = defaultdict(int)
    for inst, s in minf.items():
        cnt[len(s)] += 1
    print("MINF rollouts per instance:", dict(cnt))
    for inst, s in minf.items():
        rs = [x["reward"] for x in s]
        for x, a in zip(s, adv_from_rewards(rs, ddof)):
            x["adv"] = a

    # per-group stats
    A_groups = defaultdict(list)
    for r in rows:
        A_groups[(r["step"], r["instance"])].append(r)
    M_groups = {(s[0]["step"], inst): s for inst, s in minf.items()}

    with open(f"{OUT}/per_prompt.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "instance", "A_n", "A_k", "A_meanL", "A_L_succ", "A_L_fail", "A_contrib",
                    "M_n", "M_k", "M_meanL", "M_L_succ", "M_L_fail", "M_contrib"])
        per_step = defaultdict(lambda: dict(A=[], M=[]))
        for key in sorted(A_groups):
            ga = group_stats(A_groups[key]); gm = group_stats(M_groups[key]) if key in M_groups else None
            per_step[key[0]]["A"].append((key[1], ga))
            if gm:
                per_step[key[0]]["M"].append((key[1], gm))
            w.writerow([key[0], key[1], ga["n"], ga["k"], round(ga["meanL"]), ga["L_succ"], ga["L_fail"], ga["sumAL"],
                        *(([gm["n"], gm["k"], round(gm["meanL"]), gm["L_succ"], gm["L_fail"], gm["sumAL"]]) if gm else [None] * 6)])

    lines = []
    lines.append("| step | adv_mean A (true) | adv_mean M (recomputed) | W&B M | mixed groups A/M | pass rate A/M | L_succ A/M (mixed) | L_fail A/M (mixed) | mean L A/M |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    wb = {int(k): v for k, v in json.load(open(f"{W}/analysis/wandb_step_metrics_full.json"))["main_minf"].items()}
    wbA = {int(k): v for k, v in json.load(open(f"{W}/analysis/wandb_step_metrics_full.json"))["runA_vllm"].items()}
    paired = []
    for step in sorted(per_step):
        A = per_step[step]["A"]; M = per_step[step]["M"]
        def agg(G):
            sumAL = sum(g["sumAL"] for _, g in G); sumL = sum(g["sumL"] for _, g in G)
            mixed = [g for _, g in G if g["mixed"]]
            return dict(adv=sumAL / sumL if sumL else float("nan"), n_mixed=len(mixed),
                        pr=st.mean(g["pass_rate"] for _, g in G) if G else float("nan"),
                        Ls=st.mean(g["L_succ"] for g in mixed) if mixed else float("nan"),
                        Lf=st.mean(g["L_fail"] for g in mixed) if mixed else float("nan"),
                        mL=st.mean(g["meanL"] for _, g in G) if G else float("nan"))
        a = agg(A); m = agg(M)
        lines.append(f"| {step} | {a['adv']:+.4f} (W&B {wbA[step]['train/advantages/mean']:+.4f}) | {m['adv']:+.4f} | {wb[step]['train/advantages/mean']:+.4f} | {a['n_mixed']}/{m['n_mixed']} | {a['pr']:.3f}/{m['pr']:.3f} | {a['Ls']:.0f}/{m['Ls']:.0f} | {a['Lf']:.0f}/{m['Lf']:.0f} | {a['mL']:.0f}/{m['mL']:.0f} |")
        dm = {inst: g for inst, g in M}
        for inst, ga in A:
            if inst in dm:
                paired.append((step, inst, ga, dm[inst]))
    lines.append("")
    # paired per-prompt analysis
    dpr = [gm["pass_rate"] - ga["pass_rate"] for _, _, ga, gm in paired]
    dL = [(gm["meanL"] - ga["meanL"]) / ga["meanL"] for _, _, ga, gm in paired]
    both_mixed = [(ga, gm) for _, _, ga, gm in paired if ga["mixed"] and gm["mixed"]]
    dgap = [(gm["L_fail"] - gm["L_succ"]) - (ga["L_fail"] - ga["L_succ"]) for ga, gm in both_mixed]
    def summ(name, d):
        n = len(d); mean = st.mean(d); sd = st.stdev(d) if n > 1 else float("nan")
        lines.append(f"- {name}: n={n}, mean {mean:+.4f}, sd {sd:.4f}, t={mean / (sd / n ** 0.5):+.2f}, MINF higher in {sum(x > 0 for x in d)}/{n}")
    lines.append(f"Paired prompts: {len(paired)}")
    summ("pass rate (MINF - vLLM)", dpr)
    summ("mean generation length, relative (MINF - vLLM)/vLLM", dL)
    summ("fail-minus-success length gap (MINF - vLLM), prompts mixed in both", dgap)
    same_k = sum(1 for _, _, ga, gm in paired if ga["k"] == gm["k"])
    lines.append(f"- identical success count k in {same_k}/{len(paired)} prompts; mixed in both: {len(both_mixed)}; mixed only in vLLM: {sum(1 for _,_,ga,gm in paired if ga['mixed'] and not gm['mixed'])}; mixed only in MINF: {sum(1 for _,_,ga,gm in paired if gm['mixed'] and not ga['mixed'])}")
    # per-step paired length differences
    lines.append("")
    lines.append("| step | prompts | mean rel. length diff MINF vs vLLM | MINF longer in | pass-rate diff | prompts with same k |")
    lines.append("|---|---|---|---|---|---|")
    for step in sorted(per_step):
        P = [(ga, gm) for s, _, ga, gm in paired if s == step]
        if not P:
            continue
        d = [(gm["meanL"] - ga["meanL"]) / ga["meanL"] for ga, gm in P]
        lines.append(f"| {step} | {len(P)} | {st.mean(d):+.3f} | {sum(x > 0 for x in d)}/{len(P)} | {st.mean(gm['pass_rate'] - ga['pass_rate'] for ga, gm in P):+.3f} | {sum(1 for ga, gm in P if ga['k'] == gm['k'])} |")
    for key, g in A_groups.items():
        for x, a in zip(g, adv_from_rewards([y["reward"] for y in g])):
            x["adv_rc"] = a
    lines.append("")
    lines.append("| step | adv_mean vLLM recomputed from rewards+lengths | adv_mean MINF recomputed | diff (MINF - vLLM) |")
    lines.append("|---|---|---|---|")
    for step in sorted(per_step):
        ga = [g for k, g in A_groups.items() if k[0] == step]; gm = [g for k, g in M_groups.items() if k[0] == step]
        ra = sum(x["adv_rc"] * x["L"] for g in ga for x in g) / sum(x["L"] for g in ga for x in g)
        rm = sum(x["adv"] * x["L"] for g in gm for x in g) / sum(x["L"] for g in gm for x in g)
        lines.append(f"| {step} | {ra:+.4f} | {rm:+.4f} | {rm - ra:+.4f} |")
    lines.append("")
    lines.append("Termination of rollouts, steps 1-8 pooled (count, mean length, mean turns):")
    lines.append("")
    lines.append("| outcome / error kind | vLLM n | vLLM mean L | vLLM turns | MINF n | MINF mean L | MINF turns |")
    lines.append("|---|---|---|---|---|---|---|")
    def bucket(samples):
        out = defaultdict(list)
        for x in samples:
            out[("resolved" if x["reward"] > 0 else "failed") + " / " + str(x["err"])].append(x)
        return out
    bA = bucket([x for g in A_groups.values() for x in g]); bM = bucket([x for g in M_groups.values() for x in g])
    for k in sorted(set(bA) | set(bM)):
        a = bA.get(k, []); m = bM.get(k, [])
        fa = lambda v: f"{st.mean(v):.0f}" if v else "-"
        lines.append(f"| {k} | {len(a)} | {fa([x['L'] for x in a])} | {fa([x['turns'] for x in a if x['turns'] is not None])} | {len(m)} | {fa([x['L'] for x in m])} | {fa([x['turns'] for x in m if x['turns'] is not None])} |")
    lines.append("")
    # paired per-prompt: failures and successes separately
    pf = []; ps = []; tf = []; tsu = []
    for step, inst, ga, gm in paired:
        A = A_groups[(step, inst)]; M = M_groups[(step, inst)]
        fa = [x["L"] for x in A if x["reward"] <= 0]; fm = [x["L"] for x in M if x["reward"] <= 0]
        sa = [x["L"] for x in A if x["reward"] > 0]; sm = [x["L"] for x in M if x["reward"] > 0]
        if fa and fm:
            pf.append((st.mean(fm) - st.mean(fa)) / st.mean(fa))
            ta = [x["turns"] for x in A if x["reward"] <= 0 and x["turns"] is not None]; tm = [x["turns"] for x in M if x["reward"] <= 0]
            if ta and tm: tf.append(st.mean(tm) - st.mean(ta))
        if sa and sm:
            ps.append((st.mean(sm) - st.mean(sa)) / st.mean(sa))
            ta = [x["turns"] for x in A if x["reward"] > 0 and x["turns"] is not None]; tm = [x["turns"] for x in M if x["reward"] > 0]
            if ta and tm: tsu.append(st.mean(tm) - st.mean(ta))
    summ("failed rollouts: relative mean length (MINF - vLLM)/vLLM, prompts with failures in both", pf)
    summ("successful rollouts: relative mean length (MINF - vLLM)/vLLM, prompts with successes in both", ps)
    summ("failed rollouts: mean turns (MINF - vLLM)", tf)
    summ("successful rollouts: mean turns (MINF - vLLM)", tsu)
    report = "\n".join(lines)
    open(f"{OUT}/report.md", "w").write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
