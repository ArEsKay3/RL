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
