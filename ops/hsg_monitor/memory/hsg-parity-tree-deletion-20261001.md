---
name: hsg-parity-tree-deletion-20261001
description: "Third unattributed wipe of a live-mounted checkout: 2026-10-01 07:21 CDT, 323 tracked files vanished from the Lustre swe_vllm_parity/nemo_rl workspace on HSG while three 64-node segments had it bind-mounted; running jobs unaffected, restore from /home copy by the parity session"
metadata:
  type: project
---

2026-10-01 07:21:09-10 CDT: the nemo_rl python package (all but the two bind-mounted files megatron_worker.py and config.py), research/, AGENTS.md and one 3rdparty license vanished from /lustre/fsw/portfolios/llmservice/users/rkirby/workspaces/swe_vllm_parity/nemo_rl. examples/, ray.sub and .git stayed. Confirmed by the run manager at 07:49: nemo_rl/ held only models/ with 2 files. Same signature as the CMH wipes of 09-19 (swe_dump) and 09-24 (splice tree); see [[mounted-tree-deletion-incidents]].

Context: V-HSG-np-r2 / V-HSG2-np-r2 / V-HSG3-np-r2 segments 7585580 / 7585594 / 7587181 were running with the tree bind-mounted read-write; they kept running on the container's code plus the mounted recipe, Gym and Megatron-LM. The Eval Runner's HF export jobs also mount the tree via EXPORT_WS. The parity session was running a read-only git status at the time. Evidence: /home/rkirby/swe_vllm_parity_incidents/incident_20261001_0721_lustre_tree_deletion.txt. The run manager's scripts touch only the runs/ tree.

**How to apply:** the parity session restores additively (rsync --ignore-existing) from /home/rkirby/swe_vllm_parity and intends to chmod a-w the package and recipe dirs; the restore must land before follower 7590656 starts (~10:58 CDT, it mounts the tree at start). Keep everything pushed; never checkout/stash/reset in a live-mounted tree; treat any peer's account of who did it as a claim. Related: [[hsg-cluster-facts]], [[swe-verified-eval-vhsg-np-r2]].
