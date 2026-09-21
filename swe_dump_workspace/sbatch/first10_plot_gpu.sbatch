#!/bin/bash
#SBATCH --job-name=swe-first10-plot-gpu
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/logs/%x_%j.out
#SBATCH --error=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/logs/%x_%j.err
set -u
W=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump
CONT=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/containers/rl-gym.63635108-zstd.sqsh
date
srun --container-image=$CONT --container-mounts=$W:$W --no-container-mount-home bash -c '
set -u
PY=""
for py in /opt/nemo-rl/.venv/bin/python /usr/bin/python3 /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python $(ls -d /opt/gym_venvs/*/bin/python 2>/dev/null); do
  [ -x "$py" ] && "$py" -c "import matplotlib, numpy" 2>/dev/null && { PY=$py; break; }
done
[ -n "$PY" ] || { echo "no interpreter with matplotlib found"; exit 2; }
echo "using $PY: $($PY -c "import matplotlib, numpy; print(matplotlib.__version__, numpy.__version__)")"
cd '"$W"' && MPLCONFIGDIR=/tmp/mpl NBOOT=2000 $PY tools/first10_report.py plot; echo "EXIT $?"'
date
