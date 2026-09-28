"""Derive a run A (vLLM from scratch) evaluation config from the pinned source config: only model path, served name, output dir, caches, cluster hostname and provenance tags change.

Usage: derive_config.py SOURCE_CONFIG DEST_YAML CHECKPOINT_HF MODEL_NAME OUTPUT_DIR CACHE_ROOT [HOSTNAME]
"""
import sys

src, dst, ck, model, outdir, cache, *rest = sys.argv[1:]
hostname = rest[0] if rest else ""
s = open(src).read()
step = ck.split("step_")[1].split("/")[0]
reps = [
    ("  hostname: aws-cmh-slurm-1-login-01.nvidia.com\n", f"  hostname: '{hostname}'\n"),
    ("        model: akamehra-nano35-swe-v2-step-10\n", f"        model: {model}\n"),
    ("        efb_command: run_benchmark.py --recipe swe-bench-verified-nano-3.5 --cluster\n"
     "          aws-cmh --checkpoint /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/runs/nano35-swe-v2-stream128-inorder1-cmh-64n/checkpoints/step_10/hf\n"
     "          --model-name akamehra-nano35-swe-v2-step-10 --account nemotron_sw_post --user-path\n"
     "          /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra\n"
     "          --env-file /home/akamehra/.frontier_eval/oci-eval.env\n",
     f"        efb_command: clean replay of certified SWE-E2E config 20260907_223032_df6becfd for rkirby run A (vLLM from scratch) step_{step}; HF export via swe_dump/tools/export_checkpoint_hf_v2.sbatch\n"),
    ("  dir: /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/nemo-evaluator-rundirs/nano_v35/swebench-verified-fixed\n", f"  dir: {outdir}\n"),
    ("    model: /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/runs/nano35-swe-v2-stream128-inorder1-cmh-64n/checkpoints/step_10/hf\n", f"    model: {ck}\n"),
    ("    served_model_name: akamehra-nano35-swe-v2-step-10\n", f"    served_model_name: {model}\n"),
    ("    - /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/cache/huggingface:/cache/huggingface\n"
     "    - /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/cache/vllm:/cache/vllm\n",
     f"    - {cache}/huggingface:/cache/huggingface\n    - {cache}/vllm:/cache/vllm\n"),
]
for a, b in reps:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(dst, "w").write(s)
print(dst)
