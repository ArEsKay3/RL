# HANDOFF — SWE-E2E v2 loop-share analysis (MINF vs vLLM rollout generation)

Written 2026-09-28 12:02  by the Log Analysis session for rkirby's possible move to HSG. Everything here is reproducible from the run directories' `dumps/token_level/*.pt` chunks (and, for the main chain only, its OpenHands trajectories). Nothing in this branch is needed by any training job.

## 1. What the numbers are

**Loop-token share** of a training step = tokens inside *repetitive reasoning blocks* / all tokens generated in that step's 512 training rollouts, in percent. A reasoning block is the generated text of one assistant turn up to `</think>` (or the end of the turn). It is *repetitive* when it has at least 1,000 tokens and its UTF-8 text compresses with zlib (level 6) to under 10 % of its size; normal long blocks compress to 25-40 %, nothing falls between 10 and 20 %. A *runaway* block is the unclosed last block of a context-truncated rollout. The strict share (loop tokens / reasoning tokens) is about 1.8x the joint share and is also in `loopshare.csv`.

Every arm trains the same recipe from the same base checkpoint (`akamehra/swe_e2e_corrected/base_model/step_18/hf`) on the same 32 prompts per step (dataset order fixed; step = row // 32), so steps are paired across arms; the paired sign-flip permutation test on per-step differences is the standard significance test used in the reports.

## 2. Tools (analysis/loopfeedback)

| file | what it does |
|---|---|
| `lf_worker.py ARM EXP CHUNK.pt PARTSDIR` | labels one token-level chunk: per reasoning block n_tok, zlib ratio, class (normal / repetitive / runaway), advantage sign; writes `parts/ARM__sNNN_cKKK.{items.csv,agg.json,cycles.jsonl}`. Needs the tokenizer at the base-model HF dir and `tools/pt_numpy.py` (torch-free .pt loader). |
| `loopshare.py` | per arm/step: strict share, class-based share, repetitive-block counts, advantage-sign split -> `loopshare.csv`, `loopshare.txt`. `ARMS` list = main-table dump arms. |
| `joint_loopshare.py` | joint share (denominator = all generated tokens, cached per chunk in `joint_gen_cache.json`), adds the main chain from `mainchain/*.csv` + `mainchain_text/*.csv` via `dataset_row_map.json` -> `joint_loopshare.csv/.txt`. |
| `loopshare_svg.py` | main table: plot + Table 1 (from-scratch arms) + Table 2 (step-10 transplants) -> `loopshare_all_table.svg`; `SERIES`, `EVAL` shading, `HDR`, `TABLES` at the top. |
| `share_tables.py` | letter-free shareable versions (line chart + table): `loopshare_share_vllm_vs_minf.svg` and `..._with_cache_variants.svg`. |
| `splice_arms.json` + `splice_loopshare.py` + `splice_loopshare_svg.py` | the splice / cross-train table (`loopshare_splice_table.svg`, `splice_loopshare.csv/.txt`): registry of replay arms (kind splice / lr0 / resume; `replay_steps` or `replay_end`; `note`), replay check against the source arm, comparators from `joint_loopshare.csv`. |
| `refresh2.sbatch`, `refresh_splice.sbatch` | one-shot refresh jobs (cpu partition, 48 cpus, 200 GB): build the missing-chunk list from `jobs.txt` / `splice_arms.json`, run `lf_worker.py` in parallel, recompute, re-render. ~2 min when little is new. |
| `punct_worker.py` / `sfp_*` / `poslen_*` / `exit_*` / `lperr_*` / `lossrec_*` / `normlp.py` / `early_compare.py` / `k_compare.py` / `report*.py` | secondary analyses used in the investigation (punctuation gate, sentence-final probabilities, loop position/length, loop exits, logprob error, loss reconstruction, early-step comparisons). Each has a matching `.sbatch`. |
| `mainchain_scan.py`, `mainchain_text_scan.py`, `mainchain_report.py` | reconstruct the original MINF main chain (no token dumps) from its Gym `output.jsonl` trajectories; `mainchain*.sbatch`. |
| `tools/first10_report.py` (+ `analysis/first10/report.md`, `first10_report.pdf`) | five-arm and engine-pooled first-ten-steps comparison report. |

`jobs.txt` maps arm letter -> run directory (first two fields; one line per arm is enough). Add an arm: append a line, add it to `ARMS` in `loopshare.py` and `joint_loopshare.py`, to `NAMES/COL/DASH` in `joint_loopshare_plot.py`, and to `SERIES` / `HDR` / `TABLES` in `loopshare_svg.py`; splice arms go in `splice_arms.json` plus `SERIES` / `HDR` in `splice_loopshare_svg.py`.

## 3. Arm -> run directory map (as of this handoff)

Run directories live under `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs/`. Dumps: `dumps/rollouts/target_step_NNNNN.jsonl` (0-indexed; N = training step N+1) and `dumps/token_level/step_NNNNN_chunk_KKK.pt` (1-indexed). Job ids and rung inventories are in `analysis/run_dir_map.md` (maintained by the run manager).

| arm | description | run directory | steps with dumps |
|---|---|---|---|
| A | run A: vLLM from scratch (eval good) | `nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-20260918` | 1-44 (44 steps) |
| K | chain K: vLLM from scratch, seed 1234 (presumed good) | `nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-seed1234-20260922` | 1-28 (28 steps) |
| M | chain M: vLLM from scratch r1, chain J argument set (presumed good) | `nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r1-20260923` | 1-26 (26 steps) |
| N | chain N: vLLM from scratch r2, chain J argument set | `nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from0-nopg-r2-20260923` | 1-26 (26 steps) |
| S | chain S: vLLM from scratch, clean 9-15 stack, masking off (cancelled at step 12) | `nano35-swe-915-64n-vllm-20260925` | 1-12 (12 steps) |
| Q4 | chain Q⁗: vLLM from scratch, vLLM prefix caching OFF (final step 40) | `nano35-swe-v2-from0-noprefix-vllm-20260926` | 1-40 (40 steps) |
| Q4b | chain Q⁗2: seed 1234 replica of chain Q⁗ (on hold at step 20) | `nano35-swe-v2-from0-noprefix-vllm-seed1234-20260927` | 1-20 (20 steps) |
| G | chain G: MINF from scratch, no prefix cache (eval bad) | `nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-nopg-noprefix-20260920` | 1-36 (36 steps) |
| I | chain I: MINF from scratch, prefix cache on, replica 1 (presumed bad) | `nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r1-20260921` | 1-37 (37 steps) |
| J | chain J: MINF from scratch, prefix cache on, replica 2 (eval good) | `nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from0-prefix-nopg-r2-20260921` | 1-50 (50 steps) |
| T | chain T: MINF from scratch, clean 9-15 stack, masking off (cancelled at step 3) | `nano35-swe-915-64n-minf-20260925` | 1-3 (3 steps) |
| U | chain U: MINF from scratch, main915 stack (2026-09-27 dumps BROKEN: single-turn episodes; withheld; relaunched 2026-09-28 in a new dir) | `nano35-swe-main915-64n-minf-20260927` | 1-7 (7 steps) |
| V | chain V: MINF from scratch, vLLM-numerical-parity adapter, prefix cache kept (segment timed out at step 35) | `nano35-swe-v2-from0-parity-minf-20260927` | 1-35 (35 steps) |
| V2 | chain V2: seed 1234 replica of chain V | `nano35-swe-v2-from0-parity-minf-seed1234-20260928` | 1-5 (5 steps) |
| V3 | chain V3: seed 4321 replica of chain V | `nano35-swe-v2-from0-parity-minf-seed4321-20260928` | 1-3 (3 steps) |
| P4 | chain P⁗: MINF from scratch, prefix cache kept across refits (final step 60; eval pass@1 0.49-0.53) | `nano35-swe-v2-from0-nvshmem-keepprefix-minf-20260925` | 1-60 (60 steps) |
| P4b | chain P⁗2: seed 1234 replica of chain P⁗ (on hold at step 22) | `nano35-swe-v2-from0-nvshmem-keepprefix-minf-seed1234-20260926` | 1-22 (22 steps) |
| F | chain F: MINF from vLLM step 10 (eval good) | `nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromvllm10-20260920` | 11-35 (25 steps) |
| L | chain L: MINF from chain K step 10, prefix on, seed 1234 (presumed bad) | `nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-fromk10-prefix-nopg-20260923` | 11-31 (21 steps) |
| B | run B: MINF from MINF step 10 | `nano35-swe-v2-stream128-inorder1-cmh-64n-minf_dump-from10-20260918` | 11-29 (19 steps) |
| D | chain D: vLLM from MINF step 10 (presumed bad) | `nano35-swe-v2-stream128-inorder1-cmh-64n-vllm_dump-from10-nopg-20260919` | 11-30 (20 steps) |
| main | main chain: MINF from scratch, original (eval bad); from trajectories, no token dumps | see `mainchain*.py` | steps 2-36 |

Splice / cross-train arms (`splice_arms.json`):

| arm | design | run directory | steps with dumps |
|---|---|---|---|
| chain P | MINF engine; chain M steps 1-20 replayed, live otherwise | `nano35-swe-v2-splice-minf-runMdata-to20-20260924` | 1-32 (32 steps) |
| chain Q | vLLM engine; chain M steps 1-20 replayed, live otherwise | `nano35-swe-v2-splice-vllm-runMdata-to20-20260924` | 1-34 (34 steps) |
| chain R | vLLM engine; chain G steps 1-10 replayed, live otherwise | `nano35-swe-v2-splice-vllm-runGdata-to10-20260924` | 1-26 (26 steps) |
| chain R′ | vLLM engine; chain G steps 1-16 replayed, live otherwise | `nano35-swe-v2-splice-vllm-runGdata-to16-20260924` | 1-22 (22 steps) |
| chain P′ | MINF engine; frozen chain P step-20 weights (lr 0) | `nano35-swe-v2-fromP20-lr0-minf-20260924` | 21-35 (15 steps) |
| chain Q′ | vLLM engine; frozen chain P step-20 weights (lr 0) | `nano35-swe-v2-fromP20-lr0-vllm-20260924` | 21-36 (16 steps) |
| chain P‴ | chain P step 20 resumed with normal training (lr 3e-6, optimizer state loaded), MINF generation with the prefix cache KEPT across weight updates (invalidate_prefix_cache_on_weight_update=false), nvshmem refit | `nano35-swe-v2-fromP20-nvshmem-keepprefix-minf-20260925` | 21-50 (30 steps) |
| chain X | vLLM engine; chain G steps 1-10 replayed, live otherwise; masked replay, test: the 127 rewarded chain G rollouts that contain a >4,096-token think turn are given sample_mask 0 (rows kept in their groups) | `nano35-swe-v2-splice-vllm-runGdata-to10-maskX-rewarded-deep-20260927` | 1-31 (31 steps) |
| chain Y | vLLM engine; chain G steps 1-10 replayed, live otherwise; masked replay, control: 127 random rewarded chain G rollouts without a deep think turn are masked instead | `nano35-swe-v2-splice-vllm-runGdata-to10-maskY-random-rewarded-20260927` | no dumps |
| chain Z | vLLM engine; chain M steps 1-10 replayed, live otherwise; masked replay, mirror: chain M steps 1-10 with its 213 punished deep-think rollouts masked | `nano35-swe-v2-splice-vllm-runMdata-to10-maskZ-punished-deep-20260927` | no dumps |
| chain AA | vLLM engine; chain G steps 8-10 replayed, live otherwise; step-range ablation: only chain G steps 8-10 are replayed (unmasked); steps 1-7 and 11-25 are live vLLM | `nano35-swe-v2-splice-vllm-runGdata-8to10-20260928` | 1-10 (10 steps) |
| chain AB | vLLM engine; chain G steps 1-7 replayed, live otherwise; step-range ablation: chain G steps 1-7 replayed, live vLLM from step 8 | `nano35-swe-v2-splice-vllm-runGdata-1to7-20260928` | no dumps |

## 4. Where the results stand

Current tables (regenerated by the refresh jobs; the copies in this branch are the snapshot at handoff): `loopshare_all_table.svg` (main), `loopshare_splice_table.svg` (splice), `loopshare_share_vllm_vs_minf*.svg` (shareable), text versions `joint_loopshare.txt` and `splice_loopshare.txt`.

Findings in one paragraph each:

- **Engine tally.** MINF from scratch on the v2 stack: bad = main chain, chain G, chain I (and chain P⁗2 heading bad at step 22); clean = chain J, chain P⁗ (60 steps), chain V (35 steps). vLLM from scratch: none bad in run A, chains K, M, N, S, Q⁗, Q⁗2. Steps 1-10 never separate the arms; the verdict window is steps 20-30 (chain D reached 11-29 % there).
- **Weights carry it, the sampling engine does not.** Frozen chain P step-20 weights loop the same under MINF and vLLM (P′ 1.3 % vs Q′ 0.9 %, r = 0.95 across steps). Chain G's recorded data replayed into a vLLM run (chain R, 10 steps of G data) loops like chain G (5.2 % vs 6.0 % over 16 live steps, p 0.21) and far above the vLLM arms (p < 0.001); 16 steps of G data (chain R′) loops harder. Chain M's data replayed into MINF (chain P) stays clean.
- **Localization inside the data.** Masking the 127 rewarded chain G rollouts that contain a > 4,096-token think turn (chain X) removes nearly all of the induced looping: 1.7 % over live steps 11-31 vs chain R 5.2 % (p 0.004) and chain G 7.9 % (p < 0.001), just above the vLLM floor (0.9 %, p 0.03). Chain Y (random rewarded rows masked) is the pending control; chains AA/AB (step-range ablations) are pending.
- **Prefix cache.** Keeping the MINF prefix cache across refits (chain P‴ from P step 20, chain P⁗ from scratch) gave flat runs, but the seed replica chain P⁗2 loops (6.5 / 11.9 % at steps 21-22), so the knob is not a fix. vLLM with prefix caching off (chain Q⁗, Q⁗2) sits on the vLLM floor.
- **Loop mechanics.** Loop bodies are gradient-inert (p ≈ 0.99+ inside loops, exit token `</think>` at ~1e-4; 55-75 % of loop tokens in zero-variance groups; no overlong penalty, no KL/entropy term).

## 5. Re-running on another cluster

1. Copy or mount the run directories (only `dumps/token_level` is needed; ~10 GB/step/arm) and the base-model HF dir (tokenizer only).
2. Edit the three absolute path constants at the top of each script (`L`, `T`, `R`, `HF`) — they point at `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/loopfeedback`, `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/workspaces/swe_dump/analysis/tokens`, `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/rkirby/runs` and the HF tokenizer dir.
3. `refresh2.sbatch` / `refresh_splice.sbatch` need a CPU node with ~200 GB RAM (each `lf_worker.py` holds one chunk, up to ~1 GB); `xargs -P 40`. Adjust `#SBATCH` lines for the new scheduler. Without Slurm: `python3 lf_worker.py ARM EXP CHUNK parts` per chunk, then `python3 loopshare.py > loopshare.txt; python3 joint_loopshare.py; python3 loopshare_svg.py; python3 share_tables.py` and `python3 splice_loopshare.py; python3 splice_loopshare_svg.py`.
4. The main chain rows need its Gym trajectories (`mainchain/*.csv`, `mainchain_text/*.csv` are included here; the scanners that produced them need the Gym results dirs).
5. `parts/` (8 GB of per-chunk labels) is not in git; it is fully regenerated by the workers.

## 6. Open items

- Chain Y (control for chain X), chains AA/AB (which replayed step range carries the poison), chain Z (mirror) — all registered, awaiting node time; results land in the splice table automatically.
- Chain V seed replicas V2/V3 and chain U (main915 stack, relaunched 2026-09-28 after the prefix-skip fix): verify chain U's dump episode structure before showing it (its 2026-09-27 dumps were single-turn, reward 0).
- Eval shading calls for P⁗ (evals held 0.49-0.53) and chain V are the user's to make.
- Why some MINF-generated data (G, I, P⁗2) carries the propensity and some (J, P⁗, V) does not is still unexplained; the only separator found in steps 1-10 data is a weak within-group suppression of long/near-repetitive thinking (see the data-difference session's `analysis/datadiff`).
