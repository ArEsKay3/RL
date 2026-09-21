#!/bin/bash
#SBATCH --job-name=mpl-probe
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/logs/%x_%j.out
#SBATCH --error=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/logs/%x_%j.err
CONT=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/containers/rl-gym.63635108-zstd.sqsh
date
srun --container-image=$CONT --no-container-mount-home bash -c '
for py in /usr/bin/python3 /opt/nemo-rl/.venv/bin/python /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python $(ls -d /opt/gym_venvs/*/bin/python 2>/dev/null | head -3); do
  [ -x "$py" ] || continue
  echo "$py: $($py -c "import matplotlib, numpy; print(\"matplotlib\", matplotlib.__version__, \"numpy\", numpy.__version__)" 2>&1 | tail -1)"
done
echo "python3 in PATH: $(which python3)"; ls /opt/gym_venvs 2>/dev/null | head -5'
date
