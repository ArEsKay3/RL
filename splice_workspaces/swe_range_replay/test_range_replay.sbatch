#!/bin/bash
#SBATCH --job-name=elt-test-range-replay
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_range_replay/logs/test_range_replay_%j.out
set -u
echo "START $(date) node=$(hostname)"
srun --nodes=1 --ntasks=1 --no-container-mount-home \
  --container-image=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/containers/rl-gym.63635108-zstd.sqsh \
  --container-mounts=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_replay_splice/nemo_rl/nemo_rl:/opt/nemo-rl/nemo_rl,/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_replay_splice/nemo_rl/examples/configs:/opt/nemo-rl/examples/configs,/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_replay_splice/nemo_rl/examples/nemo_gym/nemotron-3.5-nano:/opt/nemo-rl/examples/nemo_gym/nemotron-3.5-nano,/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_replay_splice:/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_replay_splice:ro,/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_range_replay:/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_range_replay:ro,/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs:/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs:ro \
  bash -c '
    RR=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_range_replay; G=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920; M=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923
    F="^config|overrides:|^  [+a-z]|MasterConfig|foreign_rollout|resolved|DIFF|vs chain R|RESULT|Error|Traceback|File "
    for spec in "AA $G 8:10" "AB $G 1:7" "AC $M 8:10"; do set -- $spec
      echo "== config path: chain $1 (steps $3)"
      cd /opt/nemo-rl && uv run --no-sync python $RR/test_overrides_range.py /opt/nemo-rl $RR/train_cmd_chain$1.txt $2 $3 2>&1 | grep -E "$F" | sed -E "s/(API_KEY|TOKEN)=[^ ]*/\1=***/"
    done
    for spec in "$G 8 10" "$G 1 7" "$M 8 10"; do set -- $spec
      echo "== ForeignRolloutSource range semantics: $(basename $1) steps $2:$3"
      cd /opt/nemo-rl && PYTHONPATH=/opt/nemo-rl uv run --no-sync python -u $RR/test_range_replay.py $1 $2 $3 2>&1 | grep -E "range|lookup|RESULT|Error|Traceback|File "
    done
  ' 2>&1 | grep -v "pynvml\|FutureWarning\|UserWarning\|warnings.warn"
echo "END $(date)"
