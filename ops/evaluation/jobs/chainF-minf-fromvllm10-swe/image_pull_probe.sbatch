#!/bin/bash
#SBATCH --job-name=nel-image-probe
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/evaluation/jobs/chainF-minf-fromvllm10-swe/logs/%x_%j.out
#SBATCH --error=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/evaluation/jobs/chainF-minf-fromvllm10-swe/logs/%x_%j.err
date; echo "node $(hostname) user $(whoami)"
srun --container-image gitlab-master.nvidia.com/dl/joc/competitive_evaluation/nemo-evaluator-next:0.5.0.1-harbor --no-container-mount-home bash -c 'echo "eval image OK: $(cat /etc/os-release | head -1)"; python3 -c "import nemo_evaluator" 2>/dev/null && echo "nemo_evaluator importable" || echo "nemo_evaluator import not tested"' && echo "PULL_OK" || echo "PULL_FAILED"
srun --container-image /lustre/fsw/portfolios/nemotron/projects/nemotron_n3_post/eval/containers/vllm/vllm-openai-v0.26.0-ray2.56.1-arm64.sqsh --no-container-mount-home bash -c 'python3 -c "import vllm, ray; print(\"vllm\", vllm.__version__, \"ray\", ray.__version__)"' && echo "VLLM_IMAGE_OK" || echo "VLLM_IMAGE_FAILED"
date
