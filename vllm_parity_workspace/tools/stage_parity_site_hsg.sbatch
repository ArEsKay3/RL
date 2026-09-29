#!/bin/bash
# Stage the CUTLASS-DSL and quack packages the parity kernels import into
# ${WS}/parity_site, and pre-build the megatron_parity_cuda JIT extension.
#
# The Megatron policy-worker venv has nvidia-cutlass-dsl's dist-info but not the
# package, and no quack at all. The gym local_vllm_model venv in the same image
# has both at the same Python (3.13.14), torch (2.11.0+cu130) and triton (3.6.0),
# so they are copied from there rather than downloaded.
#
#   sbatch tools/stage_parity_site.sbatch
#SBATCH --job-name=vparity-stage-hsg
#SBATCH --output=/home/rkirby/swe_vllm_parity/stage-%j.out
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:30:00
set -euo pipefail

WS=/home/rkirby/swe_vllm_parity
C=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_ci/nemo_rl_ci/sqsh_files/rl-gym.64248325.sqsh

srun --container-image="$C" --container-mounts="${WS}:${WS}" --no-container-mount-home bash -lc "
set -euo pipefail
SP=/opt/gym_venvs/responses_api_models/local_vllm_model/.venv/lib/python3.13/site-packages
S=${WS}/parity_site
rm -rf \"\$S\"; mkdir -p \"\$S\"
cp -r \"\$SP/nvidia_cutlass_dsl\" \"\$SP/quack\" \"\$S/\"
for d in nvidia_cutlass_dsl-4.5.2.dist-info nvidia_cutlass_dsl_libs_base-4.5.2.dist-info \
         nvidia_cutlass_dsl_libs_cu13-4.5.2.dist-info quack_kernels-0.5.0.dist-info; do
  cp -r \"\$SP/\$d\" \"\$S/\" 2>/dev/null || true
done
printf '%s\n%s\n' '${WS}/parity_site' '${WS}/parity_site/nvidia_cutlass_dsl/python_packages' > ${WS}/parity_paths.pth
export TORCH_EXTENSIONS_DIR=${WS}/torch_extensions
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.2/bin/ptxas
mkdir -p \"\$TORCH_EXTENSIONS_DIR\"
MWSP=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.13/site-packages
cp ${WS}/parity_paths.pth \"\$MWSP/zz_parity_paths.pth\"
/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python - <<'PY'
import sys
sys.path.insert(0, '${WS}/Megatron-LM')
import cutlass, quack
print('cutlass:', cutlass.__file__)
print('quack:  ', quack.__file__)
import megatron.core.inference.parity_kernels.cute.interface  # noqa: F401
from megatron.core.inference.parity_kernels.ops import load_ops
print('load_ops:', load_ops())
PY
rm -f \"\$MWSP/zz_parity_paths.pth\"
"
