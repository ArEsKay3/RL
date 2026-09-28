"""Replay the entrypoint's config path on a rendered TRAIN_CMD token list of a
masked-replay arm and check the resolved values.

usage: test_overrides_mask.py <nemo_rl_dir> <train_cmd_tokens.txt> <expected_source_run> <expected_mask_file>
"""
import os
import sys
import traceback

nemo_rl_dir, tokens_path, exp_src, exp_mask = sys.argv[1:5]
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
    fr = getattr(mc, "foreign_rollout", None)
    print("foreign_rollout on MasterConfig:", fr)
except Exception:
    traceback.print_exc(); print("RESULT FAIL"); sys.exit(1)
fr = c.get("foreign_rollout", {})
gen = c["policy"]["generation"]
print("resolved: foreign_rollout =", fr)
print("resolved: load_replay_buffer =", c["checkpointing"].get("load_replay_buffer", "<absent>"), "| backend =", gen.get("backend"), "| grpo.seed =", c["grpo"].get("seed"), "| vllm enable_prefix_caching =", (gen.get("vllm_cfg") or {}).get("enable_prefix_caching", "<absent>"), "| overlap_param_gather =", c["policy"]["megatron_cfg"]["distributed_data_parallel_config"].get("overlap_param_gather"), "| lr =", c["policy"]["megatron_cfg"]["optimizer"]["lr"], "| dump.token_ids =", c["async_rl"]["dump"].get("token_ids", "<absent>"), "| MasterConfig dump.token_ids =", getattr(mc.async_rl.dump, "token_ids", "<absent>"))
ok = (fr.get("source_run") == exp_src and fr.get("steps") == "1:10" and fr.get("mask_file") == exp_mask
      and c["checkpointing"].get("load_replay_buffer") is False and gen.get("backend") == "vllm"
      and c["policy"]["megatron_cfg"]["distributed_data_parallel_config"].get("overlap_param_gather") is False
      and os.path.isfile(exp_mask) and c["async_rl"]["dump"].get("token_ids") == "digest" and getattr(mc.async_rl.dump, "token_ids", None) == "digest")
print("RESULT", "PASS" if ok else "FAIL(values)")
