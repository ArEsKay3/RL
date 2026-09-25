"""Look up a foreign run's own saved rollouts by (target_step, prompt_idx), for
splicing them into a live SC chain in place of a live-generated group.

Used from single_controller.py's _dispatch_one_prompt: before dispatching a
prompt to the live generation engine, check whether a foreign source run
already has data for this exact (target_step, prompt's DatumSpec.idx). If so,
skip generation/environment execution entirely and commit the foreign group's
data instead, via TQReplayBuffer.commit(..., prebuilt_train_batch_and_tags=...)
so everything downstream (sampler, _advantage_stage, training, checkpointing)
runs exactly as it does for a live commit. If not, dispatch live as today.

group_id is a fresh uuid4 per run (replay_buffer.py TQReplayBuffer.reserve),
so it is NOT a stable cross-run key -- prompt_idx is (DatumSpec.idx at dispatch
time; confirmed present as a top-level "prompt_idx" field in the source run's
own dumps/rollouts/target_step_NNNNN.jsonl records, one line per completion,
16 lines sharing one group_id per group).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Optional

import torch


class ForeignRolloutSource:
    """Index + lazy per-step tensor loader for one foreign source run."""

    def __init__(self, source_run: str, steps: range, pad_token_id: int) -> None:
        self._source_run = source_run
        self._pad_token_id = pad_token_id
        # (target_step, prompt_idx) -> group_id
        self._index: dict[tuple[int, int], str] = {}
        # step -> {sample_id: row}, populated lazily on first lookup for that step
        self._step_rows_cache: dict[int, dict[str, dict[str, Any]]] = {}
        self._build_index(steps)

    def _build_index(self, steps: range) -> None:
        """Streaming-scan each step's rollout JSONL for just (target_step,
        prompt_idx, group_id) per line -- these files run 8-14 GB each with one
        full completion record per line, so only json.loads per line + discard
        is used, never a bulk read. One line per completion; redundant across
        a group's ~16 completions, harmless (same group_id each time)."""
        for step in steps:
            # rollout dumps are 0-indexed (target_step N trains in train step
            # N+1); token-level dumps are 1-indexed by step number. Caller
            # passes 1-indexed train-step numbers in `steps`; the rollout file
            # for train step S is target_step_{S-1}.
            target_step = step - 1
            path = os.path.join(
                self._source_run, "dumps", "rollouts", f"target_step_{target_step:05d}.jsonl"
            )
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for line in f:
                    obj = json.loads(line)
                    self._index[(obj["target_step"], obj["prompt_idx"])] = obj["group_id"]

    def lookup(
        self, target_step: int, prompt_idx: int
    ) -> Optional[tuple[dict[str, torch.Tensor], list[dict[str, Any]]]]:
        """Return (train_batch, rollout_tags) for this (target_step, prompt_idx)
        if the foreign run has it, else None. train_batch matches
        record_to_train_batch's own output shape (payload.py); rollout_tags
        matches record_to_rollout_tags's shape (weight_version excluded --
        TQReplayBuffer.commit's pack_payload call sets that fresh from this
        run's own start_weight_version, a foreign stamp would be wrong here)."""
        group_id = self._index.get((target_step, prompt_idx))
        if group_id is None:
            return None
        rows = self._get_step_rows(target_step)
        members = [
            (sid, row) for sid, row in rows.items() if row["group_id"] == group_id
        ]
        if not members:
            return None
        return self._pack_group(members)

    def _get_step_rows(self, target_step: int) -> dict[str, dict[str, Any]]:
        if target_step not in self._step_rows_cache:
            # token-level dumps are 1-indexed by train step; target_step N
            # trains in train step N+1 (same convention as the rollout files).
            self._step_rows_cache[target_step] = self._load_step_rows(target_step + 1)
        return self._step_rows_cache[target_step]

    def _load_step_rows(self, step: int) -> dict[str, dict[str, Any]]:
        """Load every token_level chunk for one source (1-indexed) step.

        Row tensors are CPU, unpadded (native dump trim), keyed exactly like
        write_token_level_chunk (rollout_dump.py) wrote them: input_ids/
        token_mask trimmed from offset 0, generation_logprobs from offset 1
        (logprob_offset). Same unpad-by-input_lengths-cumsum logic already
        validated in the superseded replay_train.py's _load_step_rows.
        """
        pattern = os.path.join(
            self._source_run, "dumps", "token_level", f"step_{step:05d}_chunk_*.pt"
        )
        chunk_paths = sorted(glob.glob(pattern))
        rows: dict[str, dict[str, Any]] = {}
        for path in chunk_paths:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            lengths = obj["input_lengths"].long().reshape(-1)
            n = int(lengths.shape[0])
            ids_off = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
            lp_off = torch.cat([torch.zeros(1, dtype=torch.long), (lengths - 1).cumsum(0)])
            sample_ids = obj["sample_ids"]
            tags = obj.get("tags") or [{}] * n
            input_ids_flat = obj["input_ids"]
            token_mask_flat = obj["token_mask"]
            gen_lp_flat = obj["generation_logprobs"]
            rewards = obj["rewards"]

            for i in range(n):
                length = int(lengths[i])
                input_ids_i = input_ids_flat[ids_off[i] : ids_off[i + 1]].long()
                token_mask_i = token_mask_flat[ids_off[i] : ids_off[i + 1]].long()
                gen_lp_i = gen_lp_flat[lp_off[i] : lp_off[i + 1]].float()

                # index 0 is the offset-0 dummy/never-loss-read column (see
                # write_token_level_chunk); real per-position mask starts at
                # index 1, and add_grpo_token_loss_masks_and_generation_logprobs
                # (grpo.py) sets it to 0 for every non-generated message
                # (prompt AND any interleaved tool/user turns) and 1 only for
                # actually-sampled assistant tokens, so the first 1 is the
                # first generated token -- i.e. the end of the prompt.
                nz = (token_mask_i[1:] == 1).nonzero(as_tuple=True)[0]
                prompt_length = (int(nz[0]) + 1) if len(nz) else length

                sample_id = sample_ids[i]
                row_tags = dict(tags[i] or {})
                row_tags.pop("weight_version", None)
                rows[sample_id] = {
                    "group_id": sample_id.rsplit("_g", 1)[0],
                    "input_ids": input_ids_i,
                    "token_mask": token_mask_i,
                    "generation_logprobs": gen_lp_i,
                    "length": length,
                    "prompt_length": prompt_length,
                    "reward": float(rewards[i]),
                    "rollout_tag": row_tags,
                }
        return rows

    def _pack_group(
        self, members: list[tuple[str, dict[str, Any]]]
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
        """Build (train_batch, rollout_tags) for one group, in
        record_to_train_batch's / record_to_rollout_tags's own output shapes.
        advantages/prev_logprobs are deliberately absent: at this splice point
        (pre-generation-dispatch, i.e. the same point a live commit happens),
        neither exists yet in a real commit either -- prev_logprobs is
        recomputed fresh downstream by the trainer's own logprob-inference
        stage and advantages by the real GRPOAdvantageEstimator in
        _advantage_stage, for every group regardless of live-vs-substituted
        origin. Supplying placeholders for them here would be inert (real
        commits don't populate them either), so they're omitted.
        """
        n = len(members)
        seq_len = max(row["length"] for _, row in members)
        prompt_len = max(row["prompt_length"] for _, row in members)
        pad = self._pad_token_id

        input_ids = torch.full((n, seq_len), pad, dtype=torch.long)
        token_mask = torch.zeros((n, seq_len), dtype=torch.long)
        gen_lp = torch.zeros((n, seq_len), dtype=torch.float32)
        input_lengths = torch.zeros((n,), dtype=torch.long)
        prompt_ids_for_adv = torch.full((n, prompt_len), pad, dtype=torch.long)
        total_reward = torch.zeros((n,), dtype=torch.float32)
        # float32, not bool/long: payload.py's own canonical dtype for this
        # field (grpo.py's seq-logprob-error masking later overwrites this
        # same DataPlane field with an explicit .float() tensor, and the
        # DataPlane enforces one dtype per field across all writes).
        sample_mask = torch.ones((n,), dtype=torch.float32)

        rollout_tags: list[dict[str, Any]] = []
        for row_idx, (_sid, row) in enumerate(members):
            length = row["length"]
            input_ids[row_idx, :length] = row["input_ids"]
            token_mask[row_idx, :length] = row["token_mask"]
            gen_lp[row_idx, 1:length] = row["generation_logprobs"]
            input_lengths[row_idx] = length
            p = row["prompt_length"]
            prompt_ids_for_adv[row_idx, :p] = row["input_ids"][:p]
            total_reward[row_idx] = row["reward"]
            rollout_tags.append(row["rollout_tag"])

        train_batch = {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "generation_logprobs": gen_lp,
            "prompt_ids_for_adv": prompt_ids_for_adv,
            "total_reward": total_reward,
        }
        return train_batch, rollout_tags
