#!/bin/bash
#SBATCH --job-name=elt-test-prefix-keep
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --exclude=cpu-0003
#SBATCH --time=00:20:00
#SBATCH --comment={"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"data_loading","description":"swe_prefix_keep: validate the invalidate_prefix_cache_on_weight_update overlay (Megatron InferenceConfig + dynamic engine + NeMo RL worker) and the keep-prefix arm override set inside the run container"}}
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_prefix_keep/logs/test_prefix_keep_%j.out
set -u
SPLICE=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_replay_splice
PK=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_prefix_keep
MLM=/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
P=$PK/patches
echo "START $(date) node=$(hostname)"
srun --nodes=1 --ntasks=1 --no-container-mount-home \
  --container-image=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra/containers/rl-gym.63635108-zstd.sqsh \
  --container-mounts=$SPLICE/nemo_rl/nemo_rl:/opt/nemo-rl/nemo_rl,$SPLICE/nemo_rl/examples/configs:/opt/nemo-rl/examples/configs,$SPLICE/nemo_rl/examples/nemo_gym/nemotron-3.5-nano:/opt/nemo-rl/examples/nemo_gym/nemotron-3.5-nano,$SPLICE/Megatron-LM:$MLM,$P/nemo_rl/nemo_rl/models/generation/megatron/megatron_worker.py:/opt/nemo-rl/nemo_rl/models/generation/megatron/megatron_worker.py,$P/nemo_rl/nemo_rl/models/generation/megatron/config.py:/opt/nemo-rl/nemo_rl/models/generation/megatron/config.py,$P/Megatron-LM/megatron/core/inference/config.py:$MLM/megatron/core/inference/config.py,$P/Megatron-LM/megatron/core/inference/engines/dynamic_engine.py:$MLM/megatron/core/inference/engines/dynamic_engine.py,$SPLICE:$SPLICE:ro,$PK:$PK \
  bash -c '
    PK=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_prefix_keep
    VENVPY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
    echo "== 1. override set through NeMo RL config path (driver venv)"
    cd /opt/nemo-rl && uv run --no-sync python $PK/test_overrides_pk.py /opt/nemo-rl $PK/train_cmd_pk.txt 2>&1 | grep -E "^config|overrides:|^  [+a-z]|MasterConfig|resolved|RESULT|Error|Traceback|File " | sed -E "s/(API_KEY|TOKEN)=[^ ]*/\1=***/"
    echo "== 2. knob behaviour in the worker venv with the overlays mounted"
    cd /opt/nemo-rl && PYTHONPATH=/opt/nemo-rl $VENVPY $PK/test_prefix_keep.py $PK/patches 2>&1 | grep -E "patched|PATCHED|field|resume\(\)|forwards|FAIL|RESULT|Error|Traceback|File "
  ' 2>&1 | grep -v "pynvml\|FutureWarning\|UserWarning\|warnings.warn"
echo "END $(date)"
