"""Replay the entrypoint's config path on a rendered TRAIN_CMD token list of a
step-range replay arm and check the resolved values.

usage: test_overrides_range.py <nemo_rl_dir> <train_cmd_tokens.txt> <expected_source_run> <expected_steps>
"""
import os
import sys
import traceback

nemo_rl_dir, tokens_path, exp_src, exp_steps = sys.argv[1:5]
tokens = [t for t in open(tokens_path).read().split() if t]
entry = next(i for i, t in enumerate(tokens) if t.endswith("run_grpo_single_controller.py"))
for t in tokens[:entry]:
    if "=" in t and not t.startswith(("cd", "&&", ";", "date")):
        k, v = t.split("=", 1)
        os.environ.setdefault(k, v)
cfg_idx = tokens.index("--config")
config_path = tokens[cfg_idx + 1]
overrides = tokens[cfg_idx + 2 :]
print(f"config: {config_path}")
print(f"{len(overrides)} overrides:", *overrides, sep="\n  ")
os.chdir(nemo_rl_dir)
sys.path.insert(0, nemo_rl_dir)
from omegaconf import OmegaConf
from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers
from nemo_rl.algorithms.single_controller_utils import MasterConfig
register_omegaconf_resolvers()
try:
    cfg = load_config(config_path)
    cfg = parse_hydra_overrides(cfg, overrides)
    c = OmegaConf.to_container(cfg, resolve=True)
    mc = MasterConfig(**c)
    print("MasterConfig constructed:", type(mc).__name__)
    print("foreign_rollout on MasterConfig:", getattr(mc, "foreign_rollout", None))
except Exception:
    traceback.print_exc(); print("RESULT FAIL"); sys.exit(1)
fr = c.get("foreign_rollout", {})
gen = c["policy"]["generation"]
start, end = (int(x) for x in fr.get("steps", "").split(":"))
print("resolved: foreign_rollout =", fr, "| parsed range =", list(range(start, end + 1)))
print("resolved: load_replay_buffer =", c["checkpointing"].get("load_replay_buffer", "<absent>"), "| backend =", gen.get("backend"), "| grpo.seed =", c["grpo"].get("seed"), "| vllm enable_prefix_caching =", (gen.get("vllm_cfg") or {}).get("enable_prefix_caching", "<absent>"), "| overlap_param_gather =", c["policy"]["megatron_cfg"]["distributed_data_parallel_config"].get("overlap_param_gather"), "| lr =", c["policy"]["megatron_cfg"]["optimizer"]["lr"], "| dump.token_ids =", c["async_rl"]["dump"].get("token_ids", "<absent>"), "| mask_file =", fr.get("mask_file", "<absent>"))
import yaml
chain_r = yaml.safe_load(open("/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-v2-splice-vllm-runGdata-to10-20260924/checkpoints/step_5/config.yaml"))
mine = mc.model_dump()
RUNTIME_FILLED = ("policy.generation._mtp_weights_from_refit", "policy.generation._pad_token_id", "policy.generation.model_name",
                  "policy.generation.vllm_cfg.load_format", "policy.megatron_cfg.train_iters", "grpo.max_num_steps")
def flat(d, pre=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{pre}{k}"
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict):
            v = v[0]
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out
diffs = []
for section in ("policy", "grpo", "loss_fn", "data", "env"):
    a, b = flat(chain_r.get(section), section + "."), flat(mine.get(section), section + ".")
    for k in sorted(set(a) | set(b)):
        if k in RUNTIME_FILLED or "log_dir" in k or "results_dir" in k:
            continue
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if va != vb and not (va == "<absent>" and vb is None):
            diffs.append((k, va, vb))
print("resolved: vs chain R saved config (policy/grpo/loss_fn/data/env):", "identical" if not diffs else f"{len(diffs)} difference(s)")
for k, va, vb in diffs:
    print(f"resolved: DIFF {k}: chainR={va!r} this={vb!r}")
ok = (fr.get("source_run") == exp_src and fr.get("steps") == exp_steps and "mask_file" not in fr
      and c["checkpointing"].get("load_replay_buffer") is False and gen.get("backend") == "vllm"
      and c["policy"]["megatron_cfg"]["distributed_data_parallel_config"].get("overlap_param_gather") is False
      and c["grpo"].get("seed") == 42 and not diffs)
print("RESULT", "PASS" if ok else "FAIL(values)")
