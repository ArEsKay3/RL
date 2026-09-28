"""Aggregate analysis/deep/*.deep.json and *.deeprows.csv per engine (and per step)."""
import glob, json, os, sys
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
A = sys.argv[1]
eng = lambda r: "vllm" if "vllm_dump" in r else "minf"
acc = {}
meta = None
for f in sorted(glob.glob(os.path.join(A, "deep", "*.deep.json"))):
    j = json.load(open(f)); e = eng(j["run"]); meta = j
    for k, v in j["by"].items():
        d = acc.setdefault(e, {}).setdefault(k, {})
        for kk, arr in v.items():
            d[kk] = d.get(kk, 0) + np.array(arr, dtype=np.float64)
    for k, arr in j["tlen_hist"].items():
        d = acc[e].setdefault("tlen_" + k, 0); acc[e]["tlen_" + k] = d + np.array(arr, dtype=np.float64)
def table(kind, labels):
    rows = []
    for e in ("vllm", "minf"):
        d = acc[e][kind]; n = d["n"]
        with np.errstate(divide="ignore", invalid="ignore"):
            for i, lab in enumerate(labels):
                if n[i] == 0: continue
                rows.append({"engine": e, "bucket": lab, "tokens": int(n[i]), "share": n[i] / n.sum(),
                    "bias(prev-gen)": d["d"][i] / n[i], "abs_err": d["abs"][i] / n[i], "rms": np.sqrt(d["sq"][i] / n[i]),
                    "err>1 per1e4": 1e4 * d["gt1"][i] / n[i], "gen_lp": d["gen"][i] / n[i], "prev_lp": d["prev"][i] / n[i],
                    "w<0.2 per1e4": 1e4 * d["wlo"][i] / n[i], "w>5 per1e4": 1e4 * d["whi"][i] / n[i], "mean_w_clipped": d["w"][i] / n[i],
                    "pos_adv_share": d["pos_adv_n"][i] / n[i], "mean_w|pos": d["pos_adv_w"][i] / max(d["pos_adv_n"][i], 1), "mean_w|neg": d["neg_adv_w"][i] / max(d["neg_adv_n"][i], 1)})
    return pd.DataFrame(rows).set_index(["bucket", "engine"]).sort_index()
print("# Deep token aggregate\n")
print("## by token category"); print(table("cat", meta["cats"]).round(5).to_string())
pb = meta["pos_bins"]; print("\n## by absolute position in sequence"); print(table("pos", [f"{pb[i]}-{pb[i+1]}" for i in range(len(pb)-1)]).round(5).to_string())
tb = meta["turn_bins"]; print("\n## by turn index"); print(table("turn", [f"turn {tb[i]+1}-{tb[i+1]}" for i in range(len(tb)-1)]).round(5).to_string())
gb = meta["genlp_bins"]; print("\n## by generation logprob bucket (sampler tail)"); print(table("genlp", [f"[{gb[i]},{gb[i+1]})" for i in range(len(gb)-1)]).round(5).to_string())
print("\n## by lag (trainer_version - weight_version; 3 = unknown)"); print(table("lag", ["lag0", "lag1", "lag2+", "unknown"]).round(5).to_string())
print("\n## EOS tokens by turn index"); print(table("eos_by_turn", [f"turn {tb[i]+1}-{tb[i+1]}" for i in range(len(tb)-1)]).round(5).to_string())
print("\n## last generated token of each turn (what ended the turn)"); print(table("last_tok_of_turn", meta["cats"]).round(5).to_string())
print("\n## per-turn generated length histogram (share of turns), split by row advantage sign")
tl = meta["tlen_bins"]; labs = [f"{tl[i]}-{tl[i+1]}" for i in range(len(tl)-1)]
rows = []
for e in ("vllm", "minf"):
    for k in ("all", "pos_adv", "neg_adv", "zero_adv"):
        h = acc[e]["tlen_" + k]; tot = h.sum()
        rows.append({"engine": e, "split": k, "turns": int(tot), **{lab: h[i] / tot for i, lab in enumerate(labs)}})
print(pd.DataFrame(rows).set_index(["split", "engine"]).sort_index().round(4).to_string())
# ---- per-row
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(A, "deep", "*.deeprows.csv")))], ignore_index=True)
R["engine"] = R.run.map(eng)
print("\n## per-sequence: turn-length structure by engine and by advantage sign")
R["adv_sign"] = np.sign(R.adv).astype(int)
print(R.groupby(["engine", "adv_sign"]).agg(rows=("sample_id", "size"), n_gen=("n_gen", "mean"), n_turns=("n_turns", "mean"), mean_turn=("n_gen", lambda s: (s / R.loc[s.index, "n_turns"]).mean()),
      first_turn=("first_turn", "mean"), last_turn=("last_turn", "mean"), max_turn=("max_turn", "mean"), max_turn_p95=("max_turn", lambda s: np.percentile(s, 95)),
      eos_bias=("eos_d_mean", "mean"), eos_gen_lp=("eos_gen_mean", "mean"), last_turn_ends_eos=("last_turn_ends_eos", "mean"), w_mean=("w_mean", "mean"), frac_w_lo=("frac_w_lo", "mean"), frac_w_hi=("frac_w_hi", "mean")).round(4).to_string())
print("\n## per step: EOS bias and tail stats")
print(R.groupby(["engine", "step"]).agg(rows=("sample_id", "size"), eos_bias=("eos_d_mean", "mean"), eos_gen_lp=("eos_gen_mean", "mean"), tail_tokens_per1e4=("n_genlp_lt_m8", lambda s: 1e4 * s.sum() / R.loc[s.index, "n_gen"].sum()),
      tail_prev_minus_gen=("prev_at_tail_mean", lambda s: (s - R.loc[s.index, "gen_at_tail_mean"]).mean()), d_mean=("d_mean", "mean"), abs_mean=("abs_mean", "mean"), max_turn=("max_turn", "mean"), n_turns=("n_turns", "mean"), n_gen=("n_gen", "mean"), reward=("reward", "mean")).round(4).to_string())
print("\n## lag distribution per engine"); print(pd.crosstab(R.engine, R.lag).to_string())
print("\n## effective gradient weight: sum over tokens of sign(adv)*clip(w) vs token counts, per engine")
R["pos_tokens"] = np.where(R.adv > 0, R.n_gen, 0); R["neg_tokens"] = np.where(R.adv < 0, R.n_gen, 0)
g = R.groupby("engine").agg(pos_tokens=("pos_tokens", "sum"), neg_tokens=("neg_tokens", "sum"), eff=("eff_weight", "sum"))
g["pos_share"] = g.pos_tokens / (g.pos_tokens + g.neg_tokens); print(g.to_string())
