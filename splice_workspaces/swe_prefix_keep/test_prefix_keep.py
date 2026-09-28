"""Smoke-test the invalidate_prefix_cache_on_weight_update knob inside the run
container, with the patched files mounted where the job will see them.

Checks: the mounted files are the patched copies (md5), InferenceConfig has the
field defaulting to True, DynamicInferenceEngine.resume() bumps _weight_epoch
only when the flag is True, and NeMo RL's megatron worker forwards the key.

usage: test_prefix_keep.py <patches_dir>
"""
import hashlib
import inspect
import sys

patches = sys.argv[1]
fails = []


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


import megatron.core.inference.config as mcfg
import megatron.core.inference.engines.dynamic_engine as deng
import nemo_rl.models.generation.megatron.megatron_worker as mw
import nemo_rl.models.generation.megatron.config as nrl_cfg

for mod, rel in (
    (mcfg, "Megatron-LM/megatron/core/inference/config.py"),
    (deng, "Megatron-LM/megatron/core/inference/engines/dynamic_engine.py"),
    (mw, "nemo_rl/nemo_rl/models/generation/megatron/megatron_worker.py"),
    (nrl_cfg, "nemo_rl/nemo_rl/models/generation/megatron/config.py"),
):
    same = md5(mod.__file__) == md5(f"{patches}/{rel}")
    print(f"{'patched' if same else 'NOT PATCHED'}: {mod.__file__}")
    if not same:
        fails.append(f"overlay missing for {rel}")

field = mcfg.InferenceConfig.__dataclass_fields__.get("invalidate_prefix_cache_on_weight_update")
print("InferenceConfig field:", None if field is None else f"type={field.type} default={field.default}")
if field is None or field.default is not True:
    fails.append("InferenceConfig field missing or wrong default")

Engine = deng.DynamicInferenceEngine
for flag, expected in ((False, 0), (True, 1)):
    eng = object.__new__(Engine)
    eng.state = deng.EngineState.SUSPENDED
    eng._weight_epoch = 0
    eng.invalidate_prefix_cache_on_weight_update = flag
    try:
        eng.resume()
    except Exception as exc:
        err = type(exc).__name__
    else:
        err = "no exception"
    print(f"resume() with flag={flag}: _weight_epoch={eng._weight_epoch} (stopped by {err})")
    if eng._weight_epoch != expected:
        fails.append(f"flag={flag} epoch={eng._weight_epoch} expected {expected}")

src = inspect.getsource(Engine.__init__)
if "inference_config.invalidate_prefix_cache_on_weight_update" not in src:
    fails.append("engine __init__ does not read the flag")
wsrc = inspect.getsource(mw)
if 'mcore_generation_config.get(\n                "invalidate_prefix_cache_on_weight_update", True' not in wsrc:
    fails.append("megatron_worker does not forward the key")
print("worker forwards key:", 'invalidate_prefix_cache_on_weight_update' in wsrc,
      "| TypedDict has key:", "invalidate_prefix_cache_on_weight_update" in nrl_cfg.__dict__.get("MegatronGenerationConfig", type("x", (), {})).__annotations__
      if hasattr(nrl_cfg, "MegatronGenerationConfig") else "n/a")

for f in fails:
    print("FAIL:", f)
print("RESULT", "PASS" if not fails else "FAIL")
