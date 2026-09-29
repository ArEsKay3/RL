#!/bin/bash
# Stage the CUTLASS-DSL and quack packages the parity kernels import into
# ${WS}/parity_site from the local_vllm_model gym venv of the SAME container the
# main915 stack runs, and pre-build the megatron_parity_cuda JIT extension against
# ${WS}/Megatron-LM-parity. Prints the version match evidence first.
#
#   sbatch --partition=batch_long --qos=hero-res --reservation=sla_res_nemotron_sw_post tools/stage_parity_site.sbatch
#SBATCH --job-name=main915-parity-stage
#SBATCH --account=nemotron_sw_post
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_main915/tools/stage_parity_site.%j.out
set -euo pipefail

WS=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_main915
C=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/containers/nemo-rl_fd45cb8-67127697-gym.sqsh
MLM=${WS}/Megatron-LM-parity

srun --container-image="$C" --container-mounts="${WS}:${WS}" --no-container-mount-home bash -lc "
set -euo pipefail
MWV=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
MWSP=\$(ls -d \$MWV/lib/python3.*/site-packages | head -1)
GV=/opt/gym_venvs/responses_api_models/local_vllm_model/.venv
GSP=\$(ls -d \$GV/lib/python3.*/site-packages 2>/dev/null | head -1 || true)
echo '=== worker venv ==='; \$MWV/bin/python -c 'import sys, torch, triton; print(\"python\", sys.version.split()[0]); print(\"torch\", torch.__version__); print(\"triton\", triton.__version__)'
echo \"worker site-packages: \$MWSP\"
echo '=== worker venv: parity deps already present? ==='; \$MWV/bin/python - <<'PY'
import importlib, importlib.metadata as md
for name in ['cutlass','quack','tvm_ffi','torch_c_dlpack_ext','cuda.bindings','cuda.pathfinder','einops','ninja']:
    try:
        importlib.import_module(name); print(f'  import {name}: ok')
    except Exception as e:
        print(f'  import {name}: MISSING ({type(e).__name__})')
for dist in ['nvidia-cutlass-dsl','nvidia-cutlass-dsl-libs-cu13','quack-kernels','apache-tvm-ffi','torch-c-dlpack-ext','cuda-bindings','cuda-pathfinder','ninja']:
    try: print(f'  dist {dist}: {md.version(dist)}')
    except Exception: print(f'  dist {dist}: absent')
PY
echo '=== gym local_vllm_model venv ==='; [ -n \"\$GSP\" ] || { echo 'MISSING: local_vllm_model venv'; exit 2; }
\$GV/bin/python -c 'import sys, torch, triton; print(\"python\", sys.version.split()[0]); print(\"torch\", torch.__version__); print(\"triton\", triton.__version__)'
for d in nvidia_cutlass_dsl quack; do [ -d \"\$GSP/\$d\" ] && echo \"  present: \$GSP/\$d\" || { echo \"MISSING in gym venv: \$d\"; exit 3; }; done
ls \$GSP | grep -i -E 'cutlass|quack' | sed 's/^/  dist: /' || true
which nvcc || ls /usr/local/cuda/bin/nvcc; /usr/local/cuda/bin/nvcc --version | tail -1; ls -la /usr/local/cuda/bin/ptxas
echo '=== staging ==='
S=${WS}/parity_site
rm -rf \"\$S\"; mkdir -p \"\$S\"
cp -r \"\$GSP/nvidia_cutlass_dsl\" \"\$GSP/quack\" \"\$S/\"
for d in \$(ls \$GSP | grep -E '^(nvidia_cutlass_dsl|quack_kernels).*dist-info\$'); do cp -r \"\$GSP/\$d\" \"\$S/\"; done
ls \"\$S\"
printf '%s\n%s\n' '${WS}/parity_site' '${WS}/parity_site/nvidia_cutlass_dsl/python_packages' > ${WS}/parity_paths.pth
echo \"import os; os.environ.setdefault('TRITON_PTXAS_BLACKWELL_PATH', '/usr/local/cuda/bin/ptxas'); os.environ.setdefault('TORCH_EXTENSIONS_DIR', '${WS}/torch_extensions')\" >> ${WS}/parity_paths.pth
cat ${WS}/parity_paths.pth
export TORCH_EXTENSIONS_DIR=${WS}/torch_extensions
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
mkdir -p \"\$TORCH_EXTENSIONS_DIR\"
cp ${WS}/parity_paths.pth \"\$MWSP/zz_parity_paths.pth\"
echo '=== import + JIT build check against ${MLM} ==='
\$MWV/bin/python - <<'PY'
import sys, time
sys.path.insert(0, '${MLM}')
import cutlass, quack
print('cutlass:', cutlass.__file__)
print('quack:  ', quack.__file__)
import megatron
print('megatron:', megatron.__file__)
import megatron.core.inference.parity_kernels.cute.interface  # noqa: F401
from megatron.core.inference.parity_kernels.ops import load_ops
t=time.time(); print('load_ops:', load_ops()); print('build seconds:', round(time.time()-t,1))
import megatron.core.inference.vllm_parity as vp
print('vllm_parity module ok:', vp.__file__)
PY
rm -f \"\$MWSP/zz_parity_paths.pth\"
echo STAGING_OK
"
