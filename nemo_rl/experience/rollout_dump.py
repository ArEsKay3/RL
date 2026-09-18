# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline dumps of SingleController rollouts and per-chunk training tensors.

Enabled by ``async_rl.dump.enabled``. Two artifacts are written under the dump
root:

* ``rollouts/target_step_NNNNN.jsonl``: one line per committed completion,
  written at replay-buffer commit time with the prompt identity, reward,
  per-message structure and, when ``async_rl.dump.rollout_text`` is set, the
  decoded text via the environment's full result.
* ``token_level/step_NNNNN_chunk_CCC.pt``: one file per streaming chunk of a
  training step, written from the advantage stage with token ids, masks,
  generation/policy logprobs, advantages and rewards, trimmed per row.

Rows in both artifacts carry the DataPlane ``sample_id`` so they can be joined.
Every writer swallows its own failures: a dump problem must not end a run.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Mapping, Optional

import torch

from nemo_rl.experience.interfaces import (
    NEMO_GYM_ROLLOUT_INDEX_KEY,
    NEMO_GYM_TASK_INDEX_KEY,
    PromptGroupRecord,
)

ROLLOUTS_SUBDIR = "rollouts"
TOKEN_LEVEL_SUBDIR = "token_level"
_MAX_REPORTED_FAILURES = 5


def resolve_dump_dir(master_config: Any) -> Optional[str]:
    """Return the dump root for this run, or None when dumping is disabled."""
    dump_cfg = getattr(master_config.async_rl, "dump", None)
    if dump_cfg is None or not dump_cfg.enabled:
        return None
    if dump_cfg.dir:
        return str(dump_cfg.dir)
    logger_cfg = master_config.logger
    if isinstance(logger_cfg, Mapping):
        log_dir = logger_cfg.get("log_dir")
    else:
        log_dir = getattr(logger_cfg, "log_dir", None)
    if not log_dir:
        raise ValueError(
            "async_rl.dump.enabled=true requires async_rl.dump.dir or logger.log_dir"
        )
    return os.path.join(str(log_dir), "dumps")


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return {"__tensor__": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _numeric_scalars(mapping: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(mapping, Mapping):
        return out
    for key, value in mapping.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        out[str(key)] = float(value)
    return out


def _token_count(message: Mapping[str, Any]) -> int:
    token_ids = message.get("token_ids")
    if isinstance(token_ids, torch.Tensor):
        return int(token_ids.numel())
    if token_ids is None:
        return 0
    return len(token_ids)


def _message_summary(message: Mapping[str, Any], include_text: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "role": message.get("role"),
        "n_tokens": _token_count(message),
    }
    for flag in ("is_invalid_tool_call", "has_malformed_thinking"):
        if flag in message:
            summary[flag] = bool(message[flag])
    if include_text:
        content = message.get("content")
        if isinstance(content, str) and content:
            summary["content"] = content
    return summary


class RolloutDumpWriter:
    """Append one JSON line per committed completion."""

    def __init__(self, dump_dir: str, *, include_text: bool = True) -> None:
        self._dir = os.path.join(dump_dir, ROLLOUTS_SUBDIR)
        self._include_text = include_text
        self._failures = 0
        os.makedirs(self._dir, exist_ok=True)

    @property
    def directory(self) -> str:
        return self._dir

    def write_group(
        self,
        record: PromptGroupRecord,
        *,
        group_id: str,
        journal_id: Optional[str],
        sample_ids: list[str],
        target_step: Optional[int],
        start_weight_version: int,
        end_weight_version: int,
    ) -> None:
        try:
            self._write_group(
                record,
                group_id=group_id,
                journal_id=journal_id,
                sample_ids=sample_ids,
                target_step=target_step,
                start_weight_version=start_weight_version,
                end_weight_version=end_weight_version,
            )
        except Exception as error:  # noqa: BLE001 - diagnostics must never fail the run
            self._failures += 1
            if self._failures <= _MAX_REPORTED_FAILURES:
                print(
                    f"▶ [rollout-dump] FAILED for group {group_id}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )

    def _write_group(
        self,
        record: PromptGroupRecord,
        *,
        group_id: str,
        journal_id: Optional[str],
        sample_ids: list[str],
        target_step: Optional[int],
        start_weight_version: int,
        end_weight_version: int,
    ) -> None:
        stamp = f"{int(target_step):05d}" if isinstance(target_step, int) else "unstamped"
        path = os.path.join(self._dir, f"target_step_{stamp}.jsonl")

        prompt_extra = (
            record.extra_env_info if isinstance(record.extra_env_info, Mapping) else {}
        )
        create_params = prompt_extra.get("responses_create_params")
        prompt_metadata = (
            create_params.get("metadata") if isinstance(create_params, Mapping) else None
        )
        agent_ref = prompt_extra.get("agent_ref")
        environment = agent_ref.get("name") if isinstance(agent_ref, Mapping) else None
        task_name = (
            record.metadata.get("task_name")
            if isinstance(record.metadata, Mapping)
            else None
        )
        prompt_messages = [
            _message_summary(message, self._include_text) for message in record.prompt
        ]
        rollout_metrics = _numeric_scalars(record.rollout_metrics)
        committed_at = time.time()

        lines: list[str] = []
        for index, completion in enumerate(record.completions):
            env_extras = (
                completion.env_extras if isinstance(completion.env_extras, Mapping) else {}
            )
            messages = [
                _message_summary(message, self._include_text)
                for message in completion.message_log
            ]
            row: dict[str, Any] = {
                "sample_id": sample_ids[index] if index < len(sample_ids) else None,
                "group_id": group_id,
                "journal_id": journal_id,
                "completion_index": index,
                "target_step": target_step,
                "start_weight_version": start_weight_version,
                "end_weight_version": end_weight_version,
                "committed_at": committed_at,
                "prompt_idx": record.prompt_idx,
                "task_name": task_name,
                "environment": environment,
                "prompt_metadata": prompt_metadata,
                "nemo_gym_task_index": env_extras.get(NEMO_GYM_TASK_INDEX_KEY),
                "nemo_gym_rollout_index": env_extras.get(NEMO_GYM_ROLLOUT_INDEX_KEY),
                "reward": float(completion.reward),
                "truncated": bool(completion.truncated),
                "num_assistant_turns": sum(
                    1 for message in completion.message_log if message.get("role") == "assistant"
                ),
                "generation_length": sum(
                    _token_count(message)
                    for message in completion.message_log
                    if message.get("role") == "assistant"
                ),
                "total_tokens": sum(_token_count(message) for message in completion.message_log),
                "messages": messages,
                "rollout_metrics": rollout_metrics,
                "env_extras_numeric": _numeric_scalars(env_extras),
            }
            if self._include_text:
                row["prompt_messages"] = prompt_messages
                row["full_result"] = env_extras
            lines.append(json.dumps(row, default=_json_default, ensure_ascii=False))

        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")


def _trim_rows(tensor: torch.Tensor, lengths: torch.Tensor, offset: int) -> torch.Tensor:
    tensor = tensor.detach().to("cpu")
    pieces = [tensor[i, offset : int(lengths[i])] for i in range(int(lengths.shape[0]))]
    if not pieces:
        return tensor.new_zeros(0)
    return torch.cat(pieces)


def write_token_level_chunk(
    dump_dir: str,
    *,
    step: int,
    chunk_index: int,
    trainer_version: int,
    sample_ids: list[str],
    tags: Optional[list[dict[str, Any]]],
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask_before: torch.Tensor,
    sample_mask_after: torch.Tensor,
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    generation_logprobs: Optional[torch.Tensor] = None,
    prev_logprobs: Optional[torch.Tensor] = None,
    reference_logprobs: Optional[torch.Tensor] = None,
) -> None:
    """Write one streaming chunk's per-token training data.

    ``input_ids`` and ``token_mask`` are trimmed to each row's ``input_lengths``
    and concatenated flat. Logprob and advantage arrays start at token position 1
    (``logprob_offset``), matching how the loss indexes them, so each row
    contributes ``input_lengths[i] - 1`` values there.
    """
    try:
        out_dir = os.path.join(dump_dir, TOKEN_LEVEL_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)
        lengths = input_lengths.detach().to("cpu").long().reshape(-1)
        payload: dict[str, Any] = {
            "step": step,
            "chunk_index": chunk_index,
            "trainer_version": trainer_version,
            "sample_ids": list(sample_ids),
            "tags": [dict(tag) for tag in (tags or [])],
            "input_lengths": lengths.to(torch.int32),
            "input_ids": _trim_rows(input_ids, lengths, 0).to(torch.int32),
            "token_mask": _trim_rows(token_mask, lengths, 0).to(torch.uint8),
            "logprob_offset": 1,
            "advantages": _trim_rows(advantages, lengths, 1).float(),
            "rewards": rewards.detach().float().cpu(),
            "sample_mask_before": sample_mask_before.detach().float().cpu(),
            "sample_mask_after": sample_mask_after.detach().float().cpu(),
        }
        if generation_logprobs is not None:
            payload["generation_logprobs"] = _trim_rows(
                generation_logprobs, lengths, 1
            ).float()
        if prev_logprobs is not None:
            payload["prev_logprobs"] = _trim_rows(prev_logprobs, lengths, 1).float()
        if reference_logprobs is not None:
            payload["reference_policy_logprobs"] = _trim_rows(
                reference_logprobs, lengths, 1
            ).float()
        if generation_logprobs is not None and prev_logprobs is not None:
            mask = (
                token_mask[:, 1:].detach().float().cpu()
                * sample_mask_before.detach().float().cpu().unsqueeze(-1)
            )
            lp_error = (
                generation_logprobs[:, 1:].detach().float().cpu()
                - prev_logprobs[:, 1:].detach().float().cpu()
            ).abs()
            denom = mask.sum(dim=-1).clamp(min=1.0)
            payload["seq_mult_prob_error"] = (
                torch.exp(lp_error * mask) * mask
            ).sum(dim=-1) / denom

        path = os.path.join(out_dir, f"step_{step:05d}_chunk_{chunk_index:03d}.pt")
        tmp_path = f"{path}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        print(
            f"▶ [token-dump] step={step} chunk={chunk_index} rows={int(lengths.shape[0])} "
            f"tokens={int(lengths.sum())} -> {path}",
            flush=True,
        )
    except Exception as error:  # noqa: BLE001 - diagnostics must never fail the run
        print(
            f"▶ [token-dump] FAILED step={step} chunk={chunk_index}: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
