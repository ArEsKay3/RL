#!/bin/bash
#SBATCH --job-name=elt-test-mask-replay
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_mask_replay/logs/test_mask_replay_%j.out
set -u
SPLICE=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_replay_splice
MR=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_mask_replay
RUNS=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs
G=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920
echo "START $(date) node=$(hostname)"
srun --nodes=1 --ntasks=1 --no-container-mount-home \
  --container-image=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/containers/rl-gym.63635108-zstd.sqsh \
  --container-mounts=$SPLICE/nemo_rl/nemo_rl:/opt/nemo-rl/nemo_rl,$SPLICE/nemo_rl/examples/configs:/opt/nemo-rl/examples/configs,$SPLICE/nemo_rl/examples/nemo_gym/nemotron-3.5-nano:/opt/nemo-rl/examples/nemo_gym/nemotron-3.5-nano,$MR/patches/nemo_rl/nemo_rl/experience/foreign_rollout_source.py:/opt/nemo-rl/nemo_rl/experience/foreign_rollout_source.py,$MR/patches/nemo_rl/nemo_rl/algorithms/single_controller.py:/opt/nemo-rl/nemo_rl/algorithms/single_controller.py,$MR/patches/nemo_rl/nemo_rl/experience/rollout_dump.py:/opt/nemo-rl/nemo_rl/experience/rollout_dump.py,$MR/patches/nemo_rl/nemo_rl/algorithms/single_controller_utils/config.py:/opt/nemo-rl/nemo_rl/algorithms/single_controller_utils/config.py,$MR/patches/nemo_rl/nemo_rl/algorithms/single_controller_utils/setup.py:/opt/nemo-rl/nemo_rl/algorithms/single_controller_utils/setup.py,$SPLICE:$SPLICE:ro,$MR:$MR:ro,$RUNS:$RUNS:ro \
  bash -c '
    MR=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_mask_replay
    G=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920
    echo "== 1. override set through the NeMo RL config path (driver venv)"
    cd /opt/nemo-rl && uv run --no-sync python $MR/test_overrides_mask.py /opt/nemo-rl $MR/train_cmd_chainX.txt $G $MR/masks/poison_G_rewarded_deep_steps1-10.csv 2>&1 | grep -E "^config|overrides:|^  [+a-z]|MasterConfig|foreign_rollout|resolved|token_ids|RESULT|Error|Traceback|File " | sed -E "s/(API_KEY|TOKEN)=[^ ]*/\1=***/"
    echo "== 2. mask semantics with the overlay mounted (driver venv), target_step 0 of chain G"
    cd /opt/nemo-rl && PYTHONPATH=/opt/nemo-rl uv run --no-sync python -u $MR/test_mask_replay.py $MR/patches $MR/masks/poison_G_rewarded_deep_steps1-10.csv $G 0 2>&1 | grep -E "patched|PATCHED|mask csv|target_step|group |step 0|seq-error|estimator|rollout_dump|RolloutDumpWriter|FAIL|RESULT|Error|Traceback|File |foreign_rollout:"
  ' 2>&1 | grep -v "pynvml\|FutureWarning\|UserWarning\|warnings.warn"
echo "END $(date)"
