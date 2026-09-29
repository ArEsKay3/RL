"""Derive an evaluation config from the certified oci-hsg source config: only hostname, model path, served name, output dir, cache mounts and provenance tags change.

Usage: derive_config.py SOURCE_CONFIG DEST_YAML CHECKPOINT_HF MODEL_NAME OUTPUT_DIR CACHE_ROOT ARM [HOSTNAME]
"""
import sys

src, dst, ck, model, outdir, cache, arm, *rest = sys.argv[1:]
hostname = rest[0] if rest else ""
s = open(src).read()
step = ck.split("step_")[1].split("/")[0]
K = "/lustre/fsw/portfolios/llmservice/users/ksanthanam"
OLD_MODEL = "ksanthanam-v2-sc-minf-strict-2-step-90"
OLD_CK = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/rkirby/runs/rkirby-v2-sc-minf-strict-2/step_90/hf"
reps = [
    ("  hostname: oci-hsg-cs-001-login-01.nvidia.com\n", f"  hostname: '{hostname}'\n"),
    (f"        model: {OLD_MODEL}\n", f"        model: {model}\n"),
    ("        efb_command: run_benchmark.py --recipe swe-bench-verified-nano-3.5 --override\n"
     "          cluster.walltime=08:00:00 --override cluster.node_pools.gpu.partition=batch_long\n"
     f"          --cluster oci-hsg --account nemotron_sw_post --user-path {K}\n"
     f"          --checkpoint {OLD_CK}\n"
     f"          --model-name {OLD_MODEL}\n",
     f"        efb_command: clean replay of certified oci-hsg SWE-Bench Verified config 20260918_115143_3acfec8b (commit 81337a5b, toolchain 569ff7e8) for rkirby {arm} step_{step}\n"),
    (f"  dir: {K}/nemo-evaluator-rundirs/nano_v35/swebench-verified-fixed\n", f"  dir: {outdir}\n"),
    (f"    model: {OLD_CK}\n", f"    model: {ck}\n"),
    (f"    served_model_name: {OLD_MODEL}\n", f"    served_model_name: {model}\n"),
    (f"    - {K}/cache/huggingface:/cache/huggingface\n    - {K}/cache/vllm:/cache/vllm\n",
     f"    - {cache}/huggingface:/cache/huggingface\n    - {cache}/vllm:/cache/vllm\n"),
]
for a, b in reps:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
s = "\n".join(l for l in s.split("\n") if not l.startswith("# ")) if s.startswith("# ") else s
assert "ksanthanam" not in s, "leftover ksanthanam reference"
open(dst, "w").write(s)
print(dst)
