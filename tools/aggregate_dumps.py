"""Per-engine aggregate of analysis/rollouts/*.summary.jsonl and analysis/tokens/*.rows.csv.

Usage: aggregate_dumps.py ANALYSIS_DIR [TOKENIZER_JSON]
Prints a report and writes ANALYSIS_DIR/aggregate_report.md.
"""

from __future__ import annotations

import glob
import io
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)

A = sys.argv[1]
TOK = sys.argv[2] if len(sys.argv) > 2 else None
OUT = io.StringIO()


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.write(s + "\n")


def engine(run):
    return "vllm" if "vllm_dump" in run else "minf"


def q(s, p):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(np.percentile(s, p)) if len(s) else float("nan")


def load_tokens_map():
    if not TOK or not os.path.exists(TOK):
        return {}
    with open(TOK) as fh:
        j = json.load(fh)
    return {t["id"]: t["content"] for t in j.get("added_tokens", [])}


ID2TOK = load_tokens_map()

# ------------------------------------------------------------------ rollouts
rows = []
for f in sorted(glob.glob(os.path.join(A, "rollouts", "*.summary.jsonl"))):
    with open(f) as fh:
        rows.extend(json.loads(l) for l in fh if l.strip())
R = pd.DataFrame(rows)
if "parse_error" in R:
    P("parse errors:", int(R["parse_error"].notna().sum()))
    R = R[R["parse_error"].isna()].copy()
R["engine"] = R["run"].map(engine)
for c in ["reward", "turns", "gen_len", "total_tokens", "prompt_tokens", "max_asst_tokens", "n_invalid_tool_call",
          "n_malformed_thinking", "n_gen_without_im_end", "n_gen_without_think_close", "max_gen_chars", "n_message",
          "n_message_nonblank", "n_reasoning", "n_function_call", "n_finish", "usage_cache_read_tokens",
          "usage_prompt_tokens", "usage_completion_tokens", "latency_max", "latency_sum", "openhands_run_time",
          "total_model_call_time", "total_command_exec_time", "final_eval_time", "agent_max_turns", "model_patch_len",
          "sum_user_tokens", "n_latencies"]:
    if c in R:
        R[c] = pd.to_numeric(R[c], errors="coerce")
R["at_max_turns"] = R["turns"] >= R["agent_max_turns"]
R["ctx_hit"] = R["truncated"].astype(bool)

P("# Dump aggregate", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
P("\n## Coverage")
cov = R.groupby("engine").agg(rows=("sample_id", "size"), steps=("target_step", lambda s: f"{s.min()}-{s.max()} ({s.nunique()})"),
                              groups=("group_id", "nunique"), instances=("instance_id", "nunique"),
                              wv=("start_wv", lambda s: f"{s.min()}-{s.max()}"))
P(cov.to_string())
P("\nrows per target_step:")
P(R.pivot_table(index="target_step", columns="engine", values="sample_id", aggfunc="size", fill_value=0).T.to_string())
P("\nOverlap of instance_ids between engines:", len(set(R[R.engine == 'vllm'].instance_id) & set(R[R.engine == 'minf'].instance_id)))


def stats(g):
    d = {}
    d["rows"] = len(g)
    d["reward_mean"] = g.reward.mean()
    d["resolved_rate"] = g.resolved.astype(float).mean()
    d["patch_exists_rate"] = g.patch_exists.astype(float).mean()
    d["patch_len_p50"] = g.model_patch_len.median()
    d["truncated_rate"] = g.ctx_hit.mean()
    d["at_max_turns_rate"] = g.at_max_turns.mean()
    d["agent_max_turns"] = ",".join(str(int(x)) for x in sorted(g.agent_max_turns.dropna().unique()))
    d["turns_mean"] = g.turns.mean()
    d["turns_p50"] = g.turns.median()
    d["turns_p99"] = q(g.turns, 99)
    d["turns_max"] = g.turns.max()
    d["gen_len_mean"] = g.gen_len.mean()
    d["gen_len_p99"] = q(g.gen_len, 99)
    d["gen_len_max"] = g.gen_len.max()
    d["total_tokens_p50"] = g.total_tokens.median()
    d["total_tokens_max"] = g.total_tokens.max()
    d["prompt_tokens_mean"] = g.prompt_tokens.mean()
    d["max_asst_tokens_p99"] = q(g.max_asst_tokens, 99)
    d["max_asst_tokens_max"] = g.max_asst_tokens.max()
    d["invalid_tool_call_rows"] = (g.n_invalid_tool_call > 0).mean()
    d["invalid_tool_call_sum"] = g.n_invalid_tool_call.sum()
    d["malformed_thinking_rows"] = (g.n_malformed_thinking > 0).mean()
    d["malformed_thinking_sum"] = g.n_malformed_thinking.sum()
    d["finish_called_rate"] = (g.n_finish > 0).mean()
    d["last_gen_has_tool_call"] = g.last_gen_has_tool_call.astype(float).mean()
    d["last_gen_has_im_end"] = g.last_gen_has_im_end.astype(float).mean()
    d["rows_gen_without_im_end"] = (g.n_gen_without_im_end > 0).mean()
    d["sum_gen_without_im_end"] = g.n_gen_without_im_end.sum()
    d["rows_gen_without_think_close"] = (g.n_gen_without_think_close > 0).mean()
    d["sum_gen_without_think_close"] = g.n_gen_without_think_close.sum()
    d["max_gen_chars_p99"] = q(g.max_gen_chars, 99)
    d["max_gen_chars_max"] = g.max_gen_chars.max()
    d["n_message_mean"] = g.n_message.mean()
    d["n_message_nonblank_mean"] = g.n_message_nonblank.mean()
    d["n_reasoning_mean"] = g.n_reasoning.mean()
    d["n_function_call_mean"] = g.n_function_call.mean()
    d["cache_read_tokens_mean"] = g.usage_cache_read_tokens.mean()
    d["usage_prompt_tokens_mean"] = g.usage_prompt_tokens.mean()
    d["usage_completion_tokens_mean"] = g.usage_completion_tokens.mean()
    d["latency_max_p50"] = g.latency_max.median()
    d["latency_max_p99"] = q(g.latency_max, 99)
    d["latency_max_max"] = g.latency_max.max()
    d["openhands_run_time_p50"] = g.openhands_run_time.median()
    d["openhands_run_time_p99"] = q(g.openhands_run_time, 99)
    d["model_call_time_p50"] = g.total_model_call_time.median()
    d["cmd_exec_time_p50"] = g.total_command_exec_time.median()
    d["final_eval_time_p50"] = g.final_eval_time.median()
    d["agent_timed_out"] = g.agent_timed_out.astype(float).sum()
    d["eval_timed_out"] = g.eval_timed_out.astype(float).sum()
    d["oom_killed"] = g.oom_killed.astype(float).sum()
    d["eval_oom_killed"] = g.eval_oom_killed.astype(float).sum()
    d["resp_error_rows"] = g.resp_error.notna().sum()
    d["resp_incomplete_rows"] = g.resp_incomplete.notna().sum()
    return pd.Series(d)


P("\n## Per-engine rollout stats")
S = R.groupby("engine").apply(stats).T
P(S.to_string())


def vc(col, top=8):
    P(f"\n### {col} counts")
    t = R.groupby("engine")[col].apply(lambda s: s.fillna("None").astype(str).value_counts().head(top))
    P(t.to_string())


for col in ["agent_error_kind", "mask_sample", "resp_status", "last_item_type", "last_item_name", "resp_error", "resp_incomplete"]:
    vc(col)

P("\n### item_statuses (summed)")
for e, g in R.groupby("engine"):
    c = Counter()
    for d in g.item_statuses:
        c.update(d or {})
    P(e, dict(c))

P("\n### fc_names (summed, top 12)")
for e, g in R.groupby("engine"):
    c = Counter()
    for d in g.fc_names:
        c.update(d or {})
    P(e, dict(c.most_common(12)))

P("\n## Reward attribution cross-tabs (reward mean / n)")


def xt(col):
    t = R.groupby(["engine", col]).reward.agg(["mean", "size"]).unstack(0)
    P(f"\n### reward by {col}")
    P(t.round(3).to_string())


R["error_kind"] = R.agent_error_kind.fillna("none")
for col in ["error_kind", "ctx_hit", "at_max_turns", "mask_sample", "last_item_type"]:
    xt(col)
R["invalid_tc"] = R.n_invalid_tool_call > 0
R["malformed"] = R.n_malformed_thinking > 0
R["finish"] = R.n_finish > 0
R["gen_no_im_end"] = R.n_gen_without_im_end > 0
R["gen_no_think_close"] = R.n_gen_without_think_close > 0
for col in ["invalid_tc", "malformed", "finish", "gen_no_im_end", "gen_no_think_close"]:
    xt(col)

P("\n### resolved==True but reward != 1, or resolved False but reward != 0")
bad = R[(R.resolved.astype(bool) & (R.reward != 1.0)) | (~R.resolved.astype(bool) & (R.reward != 0.0))]
P(bad.groupby("engine").size().to_string() if len(bad) else "none")

P("\n### reward when resolved but error kind set (Gym wants these masked; SC keeps them)")
t = R[R.resolved.astype(bool) & (R.error_kind != "none")].groupby(["engine", "error_kind"]).size()
P(t.to_string() if len(t) else "none")

P("\n### truncated / max-turn rows: what ended them")
t = R[R.ctx_hit | R.at_max_turns].groupby(["engine", "ctx_hit", "at_max_turns", "error_kind"]).agg(
    n=("reward", "size"), reward=("reward", "mean"), turns=("turns", "mean"), gen_len=("gen_len", "mean"),
    total_tokens=("total_tokens", "mean"), last_no_im_end=("last_gen_has_im_end", lambda s: 1 - s.astype(float).mean()))
P(t.round(3).to_string() if len(t) else "none")

P("\n## Group-level (GRPO) reward structure")
G = R.groupby(["engine", "run", "group_id"]).agg(n=("reward", "size"), r_mean=("reward", "mean"), r_std=("reward", "std"),
                                                    steps=("target_step", "nunique"))
G["kind"] = np.select([G.r_mean == 0, G.r_mean == 1], ["all_zero", "all_one"], "mixed")
t = G.groupby(["engine", "kind"]).agg(groups=("n", "size"), rows=("n", "sum"))
P(t.to_string())
P("\ngroup sizes:", G.groupby("engine").n.apply(lambda s: dict(s.value_counts().sort_index())).to_dict())
P("groups spanning >1 target_step:", G.groupby("engine").steps.apply(lambda s: int((s > 1).sum())).to_dict())

P("\n## Per-step trend")
t = R.groupby(["engine", "target_step"]).agg(n=("reward", "size"), reward=("reward", "mean"), trunc=("ctx_hit", "mean"),
                                              max_turns=("at_max_turns", "mean"), turns=("turns", "mean"),
                                              gen_len=("gen_len", "mean"), invalid_tc=("invalid_tc", "mean"),
                                              finish=("finish", "mean"), lat_p99=("latency_max", lambda s: q(s, 99)))
P(t.round(3).to_string())

# ------------------------------------------------------------------ tokens
tf = sorted(glob.glob(os.path.join(A, "tokens", "*.rows.csv")))
if tf:
    T = pd.concat([pd.read_csv(f) for f in tf], ignore_index=True)
    T["engine"] = T["run"].map(engine)
    T["masked"] = (T.sample_mask_before > 0) & (T.sample_mask_after == 0)
    P("\n## Token-level coverage")
    P(T.groupby("engine").agg(rows=("sample_id", "size"), files=("file", "nunique"),
                               steps=("file", lambda s: ",".join(sorted({x.split("_chunk")[0].replace("step_", "") for x in s}))),
                               gen_tokens=("n_gen_tokens", "sum")).to_string())

    def tstats(g):
        d = {}
        d["rows"] = len(g)
        d["reward_mean"] = g.reward.mean()
        d["n_gen_tokens_mean"] = g.n_gen_tokens.mean()
        d["n_gen_tokens_max"] = g.n_gen_tokens.max()
        d["input_length_max"] = g.input_length.max()
        d["rollout_truncated_rate"] = pd.to_numeric(g.rollout_truncated.map({True: 1, False: 0, "True": 1, "False": 0}), errors="coerce").mean()
        d["gen_len_tag_eq_n_gen"] = (pd.to_numeric(g.rollout_generation_length, errors="coerce") == g.n_gen_tokens).mean()
        d["mask_before_mean"] = g.sample_mask_before.mean()
        d["mask_after_mean"] = g.sample_mask_after.mean()
        d["masked_rows"] = int(g.masked.sum())
        d["seq_mult_err_p50"] = g.seq_mult_prob_error.median()
        d["seq_mult_err_p90"] = q(g.seq_mult_prob_error, 90)
        d["seq_mult_err_p99"] = q(g.seq_mult_prob_error, 99)
        d["seq_mult_err_max"] = g.seq_mult_prob_error.max()
        d["seq_mult_err_gt2"] = int((g.seq_mult_prob_error > 2.0).sum())
        d["lp_err_mean_mean"] = g.lp_err_mean.mean()
        d["lp_err_mean_p99"] = q(g.lp_err_mean, 99)
        d["lp_err_p99_p50"] = g.lp_err_p99.median()
        d["lp_err_max_p50"] = g.lp_err_max.median()
        d["lp_err_max_p99"] = q(g.lp_err_max, 99)
        d["lp_err_max_max"] = g.lp_err_max.max()
        d["tokens_err_gt1"] = int(g.n_err_gt1.sum())
        d["tokens_err_gt5"] = int(g.n_err_gt5.sum())
        d["rows_err_gt1"] = (g.n_err_gt1 > 0).mean()
        d["tokens_genlp_lt_m10"] = int(g.n_genlp_lt_m10.sum())
        d["gen_lp_mean_mean"] = g.gen_lp_mean.mean()
        d["prev_lp_mean_mean"] = g.prev_lp_mean.mean()
        d["gen_lp_min_min"] = g.gen_lp_min.min()
        d["exp_err_mean_mean"] = g.exp_err_mean.mean()
        d["adv_zero_rate"] = (g.adv == 0).mean()
        d["adv_min"] = g.adv.min()
        d["adv_max"] = g.adv.max()
        d["adv_std_within_row_max"] = g.adv_std_within_row.max()
        return pd.Series(d)

    P("\n## Per-engine token-level stats")
    P(T.groupby("engine").apply(tstats).T.to_string())

    P("\n### last generated token id (top 6)")
    for e, g in T.groupby("engine"):
        c = g.last_gen_token.value_counts().head(6)
        P(e, {f"{int(k)}={ID2TOK.get(int(k), '?')}": int(v) for k, v in c.items()})

    P("\n### advantage consistency within groups (adv sign vs reward - group mean)")
    T["r_gmean"] = T.groupby(["run", "group_id"]).reward.transform("mean")
    T["r_gstd"] = T.groupby(["run", "group_id"]).reward.transform("std")
    T["gsize"] = T.groupby(["run", "group_id"]).reward.transform("size")
    dev = T.reward - T.r_gmean
    ok = (np.sign(T.adv.round(6)) == np.sign(dev.round(6)))
    P(T.assign(ok=ok).groupby("engine").ok.mean().to_string())
    P("rows with adv==0 but reward != group mean:", T[(T.adv == 0) & (dev.abs() > 1e-6)].groupby("engine").size().to_dict())
    P("rows with adv!=0 but reward == group mean:", T[(T.adv != 0) & (dev.abs() <= 1e-6)].groupby("engine").size().to_dict())
    P("group size seen in token chunks:", T.groupby("engine").gsize.apply(lambda s: dict(s.value_counts().sort_index())).to_dict())

    P("\n### masked rows (sample_mask_after == 0)")
    M = T[T.masked]
    if len(M):
        J = M.merge(R[["sample_id", "agent_error_kind", "truncated", "turns", "gen_len", "n_invalid_tool_call",
                       "last_gen_has_im_end", "resolved"]], on="sample_id", how="left")
        P(J[["engine", "sample_id", "reward", "n_gen_tokens", "seq_mult_prob_error", "lp_err_max", "n_err_gt1", "agent_error_kind",
             "truncated", "turns", "resolved"]].to_string())
    else:
        P("none")

    P("\n### rows with seq_mult_prob_error > 2 or lp_err_max > 5 (top 15 by seq_mult_prob_error)")
    X = T[(T.seq_mult_prob_error > 2) | (T.lp_err_max > 5)].sort_values("seq_mult_prob_error", ascending=False).head(15)
    if len(X):
        J = X.merge(R[["sample_id", "agent_error_kind", "truncated", "turns", "n_invalid_tool_call", "n_malformed_thinking"]],
                    on="sample_id", how="left")
        P(J[["engine", "file", "reward", "n_gen_tokens", "seq_mult_prob_error", "lp_err_mean", "lp_err_max", "n_err_gt1", "n_err_gt5",
             "sample_mask_after", "truncated", "turns", "n_invalid_tool_call"]].to_string())

    P("\n### join check: token rows with a rollout summary")
    P(T.merge(R[["sample_id", "reward"]], on="sample_id", how="left", suffixes=("", "_r")).groupby("engine").apply(
        lambda g: pd.Series({"joined": g.reward_r.notna().mean(), "reward_match": (g.reward == g.reward_r).mean()})).to_string())

    P("\n## lp_error histogram (fraction of generated tokens)")
    H = {}
    for f in sorted(glob.glob(os.path.join(A, "tokens", "*.hist.json"))):
        h = json.load(open(f))
        e = engine(h["run"])
        H.setdefault(e, {"lp": np.zeros(len(h["lp_err_hist"])), "gl": np.zeros(len(h["gen_lp_hist"])), "bins": h["lp_err_bins"], "gbins": h["gen_lp_bins"]})
        H[e]["lp"] += np.array(h["lp_err_hist"])
        H[e]["gl"] += np.array(h["gen_lp_hist"])
    for e, h in H.items():
        tot = h["lp"].sum()
        P(f"\n{e}: |gen_lp - prev_lp| over {int(tot)} tokens")
        for i in range(len(h["lp"])):
            P(f"  [{h['bins'][i]:>6}, {h['bins'][i+1]:>6})  {h['lp'][i]/tot:9.6f}  {int(h['lp'][i])}")
        tot = h["gl"].sum()
        P(f"{e}: generation logprob distribution")
        for i in range(len(h["gl"])):
            P(f"  [{h['gbins'][i]:>6}, {h['gbins'][i+1]:>6})  {h['gl'][i]/tot:9.6f}  {int(h['gl'][i])}")

with open(os.path.join(A, "aggregate_report.md"), "w") as fh:
    fh.write(OUT.getvalue())
