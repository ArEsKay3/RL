#!/bin/bash
#SBATCH --job-name=extract-vllm-src
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --qos=cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/logs/%x_%j.out
#SBATCH --error=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/logs/%x_%j.err
set -x
OUT=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/vllm_container_src
srun --nodes=1 --ntasks=1 --no-container-mount-home --container-image=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/containers/rl-gym.63635108-zstd.sqsh --container-mounts=/scratch:/scratch bash -c '
PY=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
$PY -c "import vllm, sys; print(\"VLLM_VERSION\", vllm.__version__); print(\"VLLM_PATH\", vllm.__path__[0])" 2>&1 | tail -3
SP=$($PY -c "import vllm; print(vllm.__path__[0])" 2>/dev/null)
echo "SP=$SP"
for f in tool_parsers/qwen3coder_tool_parser.py tool_parsers/utils.py tool_parsers/abstract_tool_parser.py reasoning/basic_parsers.py reasoning/deepseek_r1_reasoning_parser.py reasoning/abs_reasoning_parsers.py entrypoints/openai/engine/serving.py entrypoints/openai/chat_completion/serving.py renderers/online_renderer.py renderers/params.py v1/sample/sampler.py v1/sample/ops/topk_topp_sampler.py _version.py version.py; do
  if [ -f "$SP/$f" ]; then mkdir -p "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/vllm_container_src/$(dirname $f)"; cp "$SP/$f" "/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/vllm_container_src/$f"; echo "copied $f"; else echo "missing $f"; fi
done
$PY -m pip show vllm 2>/dev/null | head -3
ls /opt/ray_venvs | head -20
'
echo "EXIT $?"
