#!/bin/bash
# Flag audit for one campaign job: audit_flags.sh <jobid> | audit_flags.sh --dir <run_dir> [jobid]
R=/lustre/fsw/portfolios/llmservice/users/rkirby/runs
if [ "$1" = "--dir" ]; then rd=$2; jid=${3:-}; exp=$(basename "$rd"); else jid=$1; exp=$(timeout 30 squeue -j "$jid" -h -o "%j" 2>/dev/null | head -1); [ -z "$exp" ] && exp=$(timeout 30 sacct -j "$jid" -n -X -o JobName%120 2>/dev/null | head -1 | tr -d ' '); rd="$R/$exp"; fi
[ -d "$rd" ] || { echo "AUDIT $jid: no run dir $rd"; exit 1; }
sj=""; [ -n "$jid" ] && sj=$(timeout 30 scontrol show job "$jid" 2>/dev/null)
g() { echo "$sj" | grep -oE "$1=[^ ]*" | head -1 | cut -d= -f2-; }
src=""; cmd=""
ld=$(ls -dt "$rd/ray_logs/${jid}-logs" "$rd/ray_logs/${jid}-"[0-9]*"-logs" 2>/dev/null | head -1)
if [ -n "$ld" ] && [ -f "$ld/driver_command.sh" ]; then src="driver_command.sh"; cmd=$(cat "$ld/driver_command.sh"); fi
if [ -z "$cmd" ]; then p=$(ls -t "$rd"/runs/*/provenance.txt 2>/dev/null | grep -v /latest/ | head -1); [ -n "$p" ] && { src="provenance $(basename "$(dirname "$p")") (submit-time render, overwritten by same-minute resubmits)"; cmd=$(grep '^command:' "$p"); }; fi
v() { echo "$cmd" | tr ' ' '\n' | grep -E "^\+?$1=" | tail -1 | cut -d= -f2-; }
echo "AUDIT job ${jid:-?} $exp"
echo "  source: ${src:-NONE (no driver_command.sh, no provenance)}"
[ -n "$sj" ] && echo "  slurm: state=$(g JobState) partition=$(g Partition) qos=$(g QOS) account=$(g Account) reservation=$(g Reservation) nodes=$(g NumNodes) wall=$(g TimeLimit) requeue=$(g Requeue) comment_reaper=$(echo "$sj" | grep -c OccupiedIdleGPUsJobReaper)"
p=$(ls -t "$rd"/runs/*/provenance.txt 2>/dev/null | grep -v /latest/ | head -1)
[ -n "$p" ] && echo "  provenance: branch=$(grep '^branch:' "$p" | cut -d' ' -f2) commit=$(grep '^commit:' "$p" | cut -d' ' -f2 | cut -c1-9) dirty=$(grep '^dirty:' "$p" | cut -d' ' -f2-) container=$(basename "$(grep '^container:' "$p" | cut -d' ' -f2)")"
cfg=$(echo "$cmd" | grep -oE -- '--config [^ ]+' | cut -d' ' -f2)
lrb=$(v checkpointing.load_replay_buffer); opg=$(v policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather)
mpc=$(v policy.generation.mcore_generation_config.enable_prefix_caching); vpc=$(v policy.generation.vllm_cfg.enable_prefix_caching)
inv=$(v policy.generation.mcore_generation_config.invalidate_prefix_cache_on_weight_update)
echo "  config=$cfg engine=$(case "$cfg" in *minf*) echo minf;; *vllm*) echo vllm;; *) echo '?';; esac)"
echo "  load_replay_buffer=${lrb:-ABSENT}  overlap_param_gather=${opg:-ABSENT(yaml default true)}  minf_prefix_caching=${mpc:-not overridden}  vllm_prefix_caching=${vpc:-absent(=on)}  invalidate_prefix_on_refit=${inv:-yaml}"
echo "  seed=$(v grpo.seed) prompts/step=$(v grpo.num_prompts_per_step) gbs=$(v policy.train_global_batch_size) min_groups=$(v async_rl.min_groups_for_streaming_train) max_buffered=$(v async_rl.max_buffered_rollouts) nodes=$(v cluster.num_nodes) gen_nodes=$(v policy.generation.colocated.resources.num_nodes) segment=$(v cluster.segment_size) max_steps=$(v grpo.max_num_steps) ckpt_enabled=$(v checkpointing.enabled)"
echo "  other overrides: $(echo "$cmd" | tr ' ' '\n' | grep -E '^\+?[a-z_]+\.[a-zA-Z_.]+=' | grep -vE '^\+?(policy.model_name|cluster\.|policy.generation.colocated.resources|env.nemo_gym|checkpointing.checkpoint_dir|data\.|sif_dir|logger\.|grpo.max_num_steps|policy.megatron_cfg.(tensor|context|expert)_model_parallel_size|grpo.num_prompts_per_step|policy.train_global_batch_size|async_rl.(min_groups|max_buffered|max_inflight|dump.dir)|checkpointing.enabled|policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather|policy.generation.mcore_generation_config.enable_prefix_caching|checkpointing.load_replay_buffer)' | tr '\n' ' ')"
ck=$(ls -d "$rd"/checkpoints/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
if [ -n "$ck" ] && [ -f "$ck/config.yaml" ]; then
  echo "  resolved $(basename "$ck")/config.yaml: $(python3 - "$ck/config.yaml" <<'PY'
import sys,yaml
c=yaml.safe_load(open(sys.argv[1]))
def get(p):
    x=c
    for k in p.split('.'):
        if not isinstance(x,dict) or k not in x: return 'absent'
        x=x[k]
    return x
print(' '.join(f"{k.split('.')[-1]}={get(k)}" for k in ['checkpointing.load_replay_buffer','policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather','policy.generation.mcore_generation_config.enable_prefix_caching','policy.generation.vllm_cfg.enable_prefix_caching','grpo.seed','grpo.num_prompts_per_step','grpo.num_generations_per_prompt','policy.train_global_batch_size','checkpointing.save_period','async_rl.sampler.name','async_rl.sampler.max_lookahead_versions','policy.generation.mcore_generation_config.refit_backend']))
PY
)"
fi
bad=""
case "$lrb" in false|False) ;; *) bad="$bad load_replay_buffer_not_false";; esac
case "$exp" in *main915*) [ "$lrb" != "false" ] && bad="$bad (main915: yaml false but INERT on SC path, resume needs rkirby OK)";; esac
case "$opg" in false|False) ;; *) bad="$bad overlap_param_gather_not_false";; esac
[ -n "$sj" ] && { [ "$(g Account)" != "nemotron_sw_post" ] && bad="$bad account=$(g Account)"; [ "$(g Reservation)" != "(null)" ] && [ -n "$(g Reservation)" ] && bad="$bad reservation=$(g Reservation)"; [ "$(echo "$sj" | grep -c OccupiedIdleGPUsJobReaper)" = 0 ] && bad="$bad no_reaper_exemption"; case "$(g Partition)" in batch|batch_long) ;; *) bad="$bad partition=$(g Partition)";; esac; }
case "$cfg" in *minf*) case "$mpc" in true|True|"") ;; *) bad="$bad minf_prefix_caching=$mpc(rule 09-24 says true)";; esac;; esac
if [ -n "$bad" ]; then echo "  VERDICT: FAIL:$bad"; else echo "  VERDICT: PASS"; fi
