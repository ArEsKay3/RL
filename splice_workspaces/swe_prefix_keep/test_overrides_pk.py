"""Replay the entrypoint's config path (load_config -> parse_hydra_overrides ->
OmegaConf.to_container(resolve=True) -> MasterConfig) on the rendered TRAIN_CMD
token list of the keep-prefix arm and check the resolved values.

usage: test_overrides_pk.py <nemo_rl_dir> <train_cmd_tokens.txt>
"""
import os
import sys
import traceback

nemo_rl_dir, tokens_path = sys.argv[1], sys.argv[2]
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
except Exception:
    traceback.print_exc()
    print("RESULT FAIL")
    sys.exit(1)

opt = c["policy"]["megatron_cfg"]["optimizer"]
gen = c["policy"]["generation"]
mg = gen["mcore_generation_config"]
print("resolved: optimizer.lr =", opt["lr"], "| min_lr =", opt["min_lr"])
print("resolved: checkpointing.save_optimizer =", c["checkpointing"].get("save_optimizer"),
      "| load_replay_buffer =", c["checkpointing"].get("load_replay_buffer", "<absent>"),
      "| checkpoint_dir =", c["checkpointing"]["checkpoint_dir"])
print("resolved: foreign_rollout present =", "foreign_rollout" in c, "| generation backend =", gen.get("backend"),
      "| overlap_param_gather =", c["policy"]["megatron_cfg"]["distributed_data_parallel_config"].get("overlap_param_gather"))
print("resolved: mcore refit_backend =", mg.get("refit_backend"), "| enable_prefix_caching =", mg.get("enable_prefix_caching"),
      "| kv_cache_management_mode =", mg.get("kv_cache_management_mode"),
      "| invalidate_prefix_cache_on_weight_update =", mg.get("invalidate_prefix_cache_on_weight_update", "<absent>"))
ok = (
    opt["lr"] == 3e-6
    and mg.get("refit_backend") == "nvshmem"
    and mg.get("enable_prefix_caching") is True
    and mg.get("kv_cache_management_mode") == "persist"
    and mg.get("invalidate_prefix_cache_on_weight_update") is False
    and c["checkpointing"].get("load_replay_buffer") is False
    and c["checkpointing"].get("save_optimizer") is True
    and "foreign_rollout" not in c
    and gen.get("backend") == "megatron"
    and c["policy"]["megatron_cfg"]["distributed_data_parallel_config"].get("overlap_param_gather") is False
)
print("RESULT", "PASS" if ok else "FAIL(values)")
