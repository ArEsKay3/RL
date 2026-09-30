# swe_main915 / chain U handoff

Written 2026-09-28 (CMH). Workspace: `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_main915/`.
Everything below the "Data" section is code or docs and lives in the git branches; data stays on Lustre.

## 1. What chain U is

SWE-agent RL (Nemotron-3.5-nano, `swe_e2e_corrected` data, in-engine Megatron inference = "MINF")
trained **from scratch on the current NeMo RL / Megatron-LM mains**, so it can be compared with the
v2-stack arms (chain V, P⁗, ...). Recipe: `examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_dump_minf.yaml`
(rollout + token-level dumps on), 64 nodes = 32 train + 32 generation, 4 h singleton-chained segments.

Submitted 2026-09-28 11:27 PDT on the regular batch queue (rkirby's clearance, no reservation):
jobs **4072307 → 4072308 → 4072310**, partition `batch`, QOS `normal`.
Run dir `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/nano35-swe-main915-64n-minf-20260928`
(`checkpoints/` auto-resumes across segments, `dumps/`, `gym_results/`, `ray_logs/<jobid>-logs/ray-driver.log`).
W&B `ultra-v3-swe-e2e-convergence / nano35-swe-main915-64n-minf-20260928`. Monitored by the "Manage Ongoing Runs" session.

## 2. Code: repos, branches, commits

| tree (local path under the workspace) | branch | HEAD | upstream base | fork to push to |
|---|---|---|---|---|
| `nemo_rl/` | `rkirby/swe-main915-latest` | `b97225bb6` (+ this handoff commit) | NeMo RL `main` `4aaa48fab` (2026-09-24) | `git@github.com:ArEsKay3/RL.git` (already `origin`) |
| `Megatron-LM/` | `mlm-main915-latest` | `475167fa4` | Megatron-LM `main` `6a366090` (2026-09-22, the commit Bridge pins) | `git@github.com:ArEsKay3/Megatron-LM.git` |
| `Megatron-LM-parity/` (separate clone, parity arm only) | `mlm-main915-parity` | `f28af974d` | `mlm-main915-latest` `475167fa4` + santhnm2 `vllm-numerical-parity-main` (`cfa2b0b48`, `0353d829c`) + rkirby's two gate fixes | `git@github.com:ArEsKay3/Megatron-LM.git` (branch `mlm-main915-parity`; add remote `fork` first) |
| `Megatron-Bridge/` | `mlm-bridge-main915-latest` | `1f8873bb0` (= upstream, no local commits) | Bridge pin of NeMo RL main | `git@github.com:ArEsKay3/Megatron-Bridge.git` |
| `nemo_rl/3rdparty/Gym-workspace/Gym/` | `rkirby/gym-main915` | `d54e6374e` | Gym `267305e2a` (NeMo RL main's pin) | `git@github.com:ArEsKay3/Gym.git` |

`Megatron-LM/` and `Megatron-Bridge/` were cloned from a job-temporary working clone
(`/home/rkirby/.claude/jobs/51d77432/tmp/mlmwork/{mlm,bridge}`, their `origin`, which will disappear);
both have `upstream` = the NVIDIA repos. `nemo_rl/` and `Megatron-LM/` used to borrow their object
stores from `workspaces/pipeline_B/.git/modules/...` via `.git/objects/info/alternates`; on 2026-09-28
they were repacked (`git repack -a -d`) and the alternates removed, so every tree here is self-contained
and can be rsynced or pushed as-is.

**Push status: NOT pushed as of this writing** — the session that prepared this could not push
(its permission policy blocks outbound git pushes). Run from the workspace on CMH:

```bash
WS=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_main915
chmod -R u+w $WS/nemo_rl/.git $WS/Megatron-LM/.git $WS/Megatron-Bridge/.git $WS/nemo_rl/3rdparty/Gym-workspace/Gym/.git
git -C $WS/nemo_rl push origin rkirby/swe-main915-latest
git -C $WS/Megatron-LM remote add fork git@github.com:ArEsKay3/Megatron-LM.git 2>/dev/null; git -C $WS/Megatron-LM push fork mlm-main915-latest
git -C $WS/Megatron-LM-parity remote add fork git@github.com:ArEsKay3/Megatron-LM.git 2>/dev/null; git -C $WS/Megatron-LM-parity push fork mlm-main915-parity
git -C $WS/Megatron-Bridge remote add fork git@github.com:ArEsKay3/Megatron-Bridge.git 2>/dev/null; git -C $WS/Megatron-Bridge push fork mlm-bridge-main915-latest
git -C $WS/nemo_rl/3rdparty/Gym-workspace/Gym remote add fork git@github.com:ArEsKay3/Gym.git 2>/dev/null; git -C $WS/nemo_rl/3rdparty/Gym-workspace/Gym push fork d54e6374e:refs/heads/rkirby/gym-main915
```
(No force needed; all four are new branch names. The trees are deliberately `chmod a-w` — the
wipe-mitigation lock — re-lock the `.git` dirs afterwards if you keep working here.)
Never `checkout`/`reset`/`stash` inside this workspace while chain U jobs are pending or running: it is
bind-mounted read-write into them (`USE_SNAPSHOT=0`).

### Rebuilding the workspace elsewhere (e.g. HSG)

```bash
git clone -b rkirby/swe-main915-latest git@github.com:ArEsKay3/RL.git nemo_rl
git -C nemo_rl submodule update --init 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge   # optional; the launcher mounts its own
git clone -b rkirby/gym-main915 git@github.com:ArEsKay3/Gym.git nemo_rl/3rdparty/Gym-workspace/Gym
git clone -b mlm-bridge-main915-latest git@github.com:ArEsKay3/Megatron-Bridge.git Megatron-Bridge
git clone -b mlm-main915-latest git@github.com:ArEsKay3/Megatron-LM.git Megatron-LM
cp nemo_rl/swe_main915_workspace/launch_swe_main915.sh .
```
The nemo_rl superproject's Gym submodule pointer records `d54e6374e`, which only exists on the fork
branch above (`.gitmodules` still names the upstream Gym URL); clone Gym explicitly as shown. The
launcher bind-mounts `nemo_rl/nemo_rl`, the recipe dir, Gym, `Megatron-Bridge` and `Megatron-LM`
(into Bridge's `3rdparty/Megatron-LM`) over the container's copies — the container's baked-in
`megatron.bridge` cannot import Megatron-LM `6a366090`, so the Bridge mount is required.
Gym's `cache/` must stay writable (OpenHands setup locks live there); the wipe-mitigation lock is
`chmod -R a-w` on `nemo_rl/nemo_rl`, the recipe dir, `Megatron-LM`, `Megatron-Bridge` and Gym's
source dirs only.

## 3. What is on top of the upstream mains (all local commits; see `git log <base>..HEAD`)

Megatron-LM (`6a366090..475167fa4`):
- `475167fa4` **prefix-skip fix**: `_compute_prefix_match` no longer applies the KV-only prefix skip
  to a hybrid model past its first chunk (`megatron/core/inference/contexts/dynamic_context.py`).
  Upstreamable as-is; bug report draft in `analysis/prefix_skip_bug/REPORT.md`.
- `9900d8be5` text-only `compact_prompt_token_ids` → `prompt_token_ids` fallback in `chat_completions.py`
  (rkirby fixed the same upstream on 09-18 as `9615cc73c`; this keeps the `.get()` variant).
- `563714f00` zero-token prefill logprob guard; detect every model EOS when splicing the prefix.
- `0c9e52648` honor all model-declared EOS tokens during generation. (`e61c9c448` = untrack a build artifact.)

NeMo RL (`4aaa48fab..`): recipe tree + offline dumps port (`3ddc636a5`), per-job Pyxis container
isolation (`e43ae3723`), Megatron generation fixes incl. selectable sampling backend (`2def61ee1`),
`nemo.lens`-less telemetry (`528be9930`), dump wiring (`ac18bcd9d`, `99e40eaba`, `02fed9278`), and the
recipe fixes needed on main: `f98f643bb` (split_validation 0), `b37d89133` (no refit_backend under
nccl_reshard), `09a391b4a` (save_data_plane/load_replay_buffer), `c0f9d3fc4` (drop Automodel-only
checkpoint fields), **`9cd702a9c` `prefix_caching_mamba_gb: 50`**, **`b97225bb6` penalties off**.

Gym (`267305e2a..d54e6374e`): three `openai_utils.py` fixes for the `openai==2.44.0` `extra="forbid"`
strictness against the vendored OpenHands/litellm client (Bedrock auth kwargs, redundant tool-message
`name`, `compact_prompt_token_ids` through the mixins).

## 4. The three things that make chain U valid (and why)

1. **Prefix-skip fix (`475167fa4`).** Every earlier main-based run had 48-58% of 150-200-turn
   episodes masked by the trainer/engine logprob check (`seq_mult_prob_error` up to 6e7). Root cause:
   on a block-aligned continuation chunk that hash-matches its KV blocks, Megatron applied the KV-only
   skip; the hybrid guard only covered the first chunk, so the Mamba layers never saw the skipped
   ~16K tokens while attention read the matched KV. Sparse, confident, copy-from-context errors past
   ~131K tokens (8 x 16384) in late-step lone prefills. Evidence: 279/334 spiked turns reported engine
   cached tokens > 0 (median 16128 = the clamp) in memory-only mode where it must be 0; solo
   256-token prefill chunks in the engine log; chain V at 0.01 spikes/1e4 tokens.
2. **`prefix_caching_mamba_gb: 50` (`9cd702a9c`).** NeMo RL main no longer defaults it (the v2 worker
   passed 50), which had put the engine in memory-only prefix caching (no Mamba cache, whole-prompt
   re-prefill every turn, and the trigger for bug 1). Same regime as chain V.
3. **Penalties off (`b97225bb6`).** The v2 arms' single-controller path never applied the -5
   invalid-tool-call / malformed-thinking advantage penalties nor the `reward_penalties` (dead code
   there; chain V shows 0 penalty metrics and resolved == reward). NeMo RL main applies both (7-13
   resolved samples per smoke step had reward zeroed). Disabled explicitly for comparability.

## 5. Smoke evidence (16 nodes, 2 steps each, all exit 0)

| smoke | config | masked seqs (of 256) | `seq_mult_prob_error` max / err>2 | 150-200-turn err>2 | spikes/1e4 at >=131K | cached-token turns |
|---|---|---|---|---|---|---|
| 4057393 (before) | memory-only, penalties on | 41 | 6.3e7 / 39 | 48.1% | 25-80 | 487 (illegal) |
| 4065835 | fix, memory-only | 1 | 3.46 / 1 | 0.0% | 0.00-0.05 | 0 |
| 4065840 | fix + mamba cache 50 GB | 0 | 1.052 / 0 | 0.0% | 0.00-0.12 | 27251/27274 (legit) |
| 4068305 | fix + mamba cache + penalties off = chain U recipe | 0 | 1.046 / 0 | 0.0% | 0.00-0.04 | 25731/25753 |

Dumps: `.../runs/nano35-swe-main915-minf-smoke-16n/dumps.4057393-prefix-skip-bug` (the bug),
`.../runs/nano35-swe-main915-minf-smoke-16n/dumps` (4065835),
`.../runs/nano35-swe-main915-minf-smoke-16n-mambacache50/dumps` (4065840),
`.../runs/nano35-swe-main915-minf-smoke-16n-nopenalty/dumps` (4068305).

## 6. Launch / resume (CMH)

```bash
cd /scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_main915
RUN_DATE=20260928 ENGINE=minf bash launch_swe_main915.sh        # adds a 4 h segment to chain U (same name = same chain, auto-resume)
ENGINE=minf RESERVATION=1 bash launch_swe_main915.sh              # reservation variant: batch_long / hero-res / 8 h
SMOKE=1 ENGINE=minf RESERVATION=1 WALLTIME=1:30:00 bash launch_swe_main915.sh   # 16-node smoke, 2 steps
DRY_RUN=1 ... renders without submitting.  EXP_NAME=... overrides the run name/dir.
```
Same-name submissions serialize via `--dependency=singleton`; checkpoints live in the run dir and
resume automatically. `ENABLE_PREFIX_CACHING=false` turns MINF prefix caching off.

## 7. What changes on HSG (verify each; nothing here was tested on HSG)

The launcher hardcodes CMH: `MINE=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby`,
`SWE_BASE=.../users/akamehra` (model `swe_e2e_corrected/base_model/step_18/hf`, data jsonl, sandbox `sif/`),
`HF_HOME=/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/rkirby/hf_home`,
`CONTAINER=$MINE/containers/nemo-rl_fd45cb8-67127697-gym.sqsh` (needs `nemo.lens`, prebaked Gym venvs,
libXrender), partition `batch` / QOS `normal` (reservation block: `batch_long` / `hero-res` /
`sla_res_nemotron_sw_post`), account `nemotron_sw_post`, and `ray.sub`'s CPUs-per-worker (CMH nodes have
140, the original HSG launch assumed 144). On HSG: re-point those paths (Lustre prefixes differ), import
the same container image, drop `RESERVATION=1`, and re-check the walltime/QOS names. The earlier
`rlvr` launcher solved the same split with a `CLUSTER=HSG|CMH` case (nemo_rl commit `f4f89e7` on the
rlvr line) — copy that pattern rather than editing defaults in place.

## 8. Health signals (per step, from the driver log / dumps)

- `num_masked_seqs_by_logprob_error` ~ 0-1 (was 17-24 with the bug); `seq_mult_prob_error` max ~1.05.
- No `malformed_think_tag_rate` / `empty_final_answer_rate` metrics (penalties off); resolved == reward.
- 0 occurrences of `compact_prompt_token_ids` errors and `Non-contiguous messages found`.
- No `Running in memory-only mode` line at engine start (means the Mamba cache took); per-turn
  `cache_read_tokens` in the rollout jsonl non-zero on nearly every turn.
- `swe_main915_workspace/analysis/prefix_skip_bug/final_validation.py <run dir>` prints all of the above from the dumps
  (torch-free; needs `swe_dump/tools/pt_numpy.py` on the path — copy it alongside if swe_dump is not available).

## 9. Data (stays on Lustre)

Runs: `$MINE/runs/nano35-swe-main915-*` (chain U + the three smokes). Persistent cache:
`$MINE/persistent_cache/<EXP_NAME>`. `gym_results/` is the disk hog (~40 GB per 2-step smoke).

## 10. vLLM numerical-parity arm on this stack (added 2026-09-29)

Purpose: Megatron in-engine inference running Keshav Santhanam's vLLM numerical-parity adapter on
the same stack, recipe, data and dumps as chain U, so plain MINF (U) vs parity MINF vs vLLM (W)
differ only in the inference path. rkirby's ask (2026-09-29 15:30): "get the changes in this branch
[santhnm2/Megatron-LM vllm-numerical-parity-main] into this and run a VLLM parity run".

**Code.**
- `Megatron-LM-parity/`: a separate local clone (the live `Megatron-LM/` that U/W bind-mount was
  never touched). Branch `mlm-main915-parity`, HEAD `f28af974d` = `mlm-main915-latest` (`475167fa4`)
  + `ddb41a01a` (cherry-pick of the adapter commit `cfa2b0b48` "Add self-contained vLLM numerical
  parity inference"; only `uv.lock` conflicted, ours kept, inert at runtime) + `ca9309086` (their docs
  commit `0353d829c`) + `05cea8670` and `f28af974d` (rkirby's two eval-mode gate fixes `517934242` /
  `d37db1077` from `rkirby/vllm-parity-armV`, still absent upstream: without them the trainer's
  `get_logprobs_presharded` pass enters the parity QKV/norm paths). The adapter's core hunks landed
  verbatim (empty interdiff against its own base `c035a426e`). Remotes in that clone: `live` (the
  local `Megatron-LM` repo), `santhnm2`, `upstream`, `oldparity` (the swe_vllm_parity tree).
- nemo_rl `rkirby/swe-main915-latest` commit `579293f2`:
  `examples/nemo_gym/nemotron-3.5-nano/swe_sc_cmh_parity_minf.yaml` (defaults
  `swe_sc_cmh_dump_minf.yaml`; `policy.megatron_cfg.model_overrides.inference_vllm_parity: true`;
  generation `expert_model_parallel_size: 1`, `expert_tensor_parallel_size: 4`, `max_tokens: 8480`;
  everything else exactly chain U, including prefix caching and `prefix_caching_mamba_gb: 50`; the
  chain V prefix-keep knob is deliberately NOT ported), plus `merged_inference_megatron_cfg`
  (`nemo_rl/models/generation/megatron/config.py`) keeping an explicit generation ETP>1 only when
  that override is set — NeMo RL main otherwise pins generation ETP to 1 and raises.
- `launch_swe_main915_parity.sh` (workspace root; copy under `nemo_rl/swe_main915_workspace/`):
  the base launcher with `MLM_TREE=Megatron-LM-parity`, the parity yaml, EXP names
  `nano35-swe-main915-64n-parity-minf-<RUN_DATE>` / `nano35-swe-main915-parity-minf-smoke-16n`, and
  a generated `parity_paths.pth` bind-mounted to
  `/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.13/site-packages/zz_parity_paths.pth`
  (adds `parity_site/` to every worker interpreter and sets `TRITON_PTXAS_BLACKWELL_PATH` and
  `TORCH_EXTENSIONS_DIR=$WS/torch_extensions`).
- `parity_site/`: `quack` 0.5.0 plus a copy of CUTLASS DSL 4.5.2, staged by
  `tools/stage_parity_site.sbatch` from THIS stack's container
  (`nemo-rl_fd45cb8-67127697-gym.sqsh`, gym venv `/opt/gym_venvs/responses_api_models/local_vllm_model/.venv`,
  py 3.13.14 / torch 2.11.0+cu130 / triton 3.6.0 = the MegatronPolicyWorker venv). That venv already
  ships cutlass-dsl 4.5.2, tvm-ffi 0.1.11, torch-c-dlpack-ext 0.1.5, cuda-bindings 13.3.1; only quack
  was missing. Never reuse `swe_vllm_parity/parity_site` (different image). Staging job 4102161:
  `STAGING_OK`, `megatron_parity_cuda` built in 35 s. On another container/cluster re-run
  `sbatch [--partition/--qos/--reservation ...] tools/stage_parity_site.sbatch` after editing its
  container path.

**Launch.**
```bash
cd $WS
SMOKE=1 RESERVATION=1 WALLTIME=2:00:00 bash launch_swe_main915_parity.sh   # 16 nodes, 2 steps
RESERVATION=1 bash launch_swe_main915_parity.sh                            # 64 nodes, 8 h singleton segment
RUN_DATE=<same date> RESERVATION=1 bash launch_swe_main915_parity.sh       # adds a segment to that arm
```

**Evidence.** Smoke 4102198 (2026-09-29 15:49, 16 nodes, reservation): the adapter is live — the
dynamic-engine log carries the parity-only graph dimensions (`[4]: 0 P + 4 D | attention
requests=N`), the startup refit ran over `nccl_reshard` with `xferdtensor_python (exact-transfer)`
(payload 61.31 GiB, EP1/ETP4 destination), rollouts served at 4-6 ms decode steps. Step-level numbers:
see the memory/handoff update at smoke close.

### 10a. Raw per-turn token-id dumps (chain AD twin, added 2026-09-29 17:45)

rkirby: "another run equivalent to AD, but dump all prompt_token_ids and generation_token_ids
faithfully as they appear on each turn" (the earlier `async_rl.dump.token_ids` modes never emitted
prompt_token_ids: `nemo_gym.py` pops them off each output item before the message log is built).

- nemo_rl `46ba369b` + `5db5017a`: `env.nemo_gym.retain_raw_token_ids: true` keeps the
  engine-reported `prompt_token_ids` / `generation_token_ids` on every trainable Gym output item
  (as int32 arrays in memory; `_build_gym_actor_config` routes the flag to the actor); they land in
  each dump row at `full_result.response.output[*].prompt_token_ids` / `.generation_token_ids`,
  beside `prompt_str` / `generation_str`. Default off; message dicts, replay buffer and batch are
  untouched. Recipe `swe_sc_cmh_parity_minf_tokids.yaml` = chain AD's recipe + that flag;
  `TOKIDS=1 bash launch_swe_main915_parity.sh` selects it (`*-tokids-*` run names).
- Verifier: `check_tokids_dump.py <run_dir>` (copy in `nemo_rl/swe_main915_workspace/tools/`):
  presence on every trainable turn, generation ids == the assistant message's `token_ids`, prompt
  ids == concatenation of all previous message `token_ids`. Smoke 4104336: 3390/3390 on all three.
- Cost: ~21 MB per rollout row (9 MB without), i.e. ~11 GB of rollout jsonl per 512-rollout step
  (prompt ids are the full prefix at every turn, ~2 M ids per 60-turn episode).
- Run: `nano35-swe-main915-64n-parity-minf-tokids-20260929`, jobs 4104754 -> 4104794 (reservation,
  8 h segments, started 17:45:56 09-29); run dir under `users/rkirby/runs/`. Label pending from the
  run manager (proposed chain AE).

**Known limits** (adapter docs + the chain V audit): the audited profile is TP4/EP1/ETP4 with prefix
caching disabled and one NVLink node; we run it with prefix caching and the Mamba prefix cache like
chain U, and with repeated train-to-generation refits, neither of which the audit covers. The
`tl.exp -> fast_exp` change in the shared SSD forward kernels is unconditional (also affects the
training forward of this arm). Report the two gate bugs upstream if not already done.
