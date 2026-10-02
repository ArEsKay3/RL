# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay recorded SWE-bench requests against a running inference-only server.

Keep each selected turn's recorded history, including tool results. Responses
are validated and saved, not executed as commands or fed into later turns.
This tests inference on realistic inputs; it does not measure SWE-bench pass@1.
"""

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def replay(
    url: str, record: dict[str, Any], output: Path, max_tokens: int
) -> dict[str, Any]:
    body = copy.deepcopy(record["body"])
    extra = body.pop("extra_body", {})
    body.update(extra)
    for key in ("api_key", "api_base", "timeout", "max_completion_tokens"):
        body.pop(key, None)
    body.update(
        model="swe-smoke",
        stream=False,
        logprobs=True,
        top_logprobs=0,
        max_tokens=max_tokens,
        return_tokenized_data=True,
        return_raw_text=True,
        # Recorded histories have text rather than NeMo-Gym token metadata.
        prevent_retokenization=False,
    )
    # Exercise the recorded greedy settings as well as stochastic filtering.
    if record["mode"] == "sampled":
        body.update(temperature=0.7, top_k=40, top_p=0.9)
    template = body.setdefault("chat_template_kwargs", {})
    template.setdefault("truncate_history_thinking", False)
    started = time.monotonic()
    with urlopen(
        Request(
            url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=900,
    ) as response:
        payload = json.load(response)
    elapsed = time.monotonic() - started
    (output / f"{record['id']}.json").write_text(
        json.dumps({"request": body, "response": payload}, indent=2) + "\n"
    )
    choice = payload["choices"][0]
    message = choice["message"]
    tokens = message["generation_token_ids"]
    logprobs = message["generation_log_probs"]
    wire_logprobs = choice["logprobs"]["content"]
    assert tokens, f"Empty generation: {record['id']}"
    assert len(tokens) == len(logprobs) == len(wire_logprobs)
    assert all(math.isfinite(lp) and lp <= 1e-5 for lp in logprobs)
    assert all(
        entry["logprob"] == max(lp, -9999.0)
        for entry, lp in zip(wire_logprobs, logprobs)
    )
    assert payload["usage"]["completion_tokens"] == len(tokens)
    assert payload["usage"]["prompt_tokens"] == len(message["prompt_token_ids"])
    assert message["role"] == "assistant"
    assert choice["finish_reason"] in ("stop", "length", "tool_calls")
    assert (
        message.get("content") or message.get("reasoning") or message.get("tool_calls")
    )
    for call in message.get("tool_calls") or []:
        json.loads(call["function"]["arguments"])
    result = {k: v for k, v in record.items() if k != "body"}
    result.update(
        latency_s=round(elapsed, 3),
        prompt_tokens=payload["usage"]["prompt_tokens"],
        completion_tokens=len(tokens),
        finish_reason=choice["finish_reason"],
        reasoning_chars=len(message.get("reasoning") or ""),
        content_chars=len(message.get("content") or ""),
        tool_calls=len(message.get("tool_calls") or []),
    )
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Server base URL ending in /v1")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", default=["0", "1", "32"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for session in args.sessions:
        path = args.traces / session / "trace.json"
        raw = path.read_bytes()
        trace = json.loads(raw)
        turns = trace["turns"]
        for index in sorted({0, len(turns) // 2, len(turns) - 1}):
            turn = turns[index]
            for mode in ("recorded", "sampled"):
                records.append(
                    dict(
                        id=f"{session}-turn{turn['turn']}-{mode}",
                        session=trace["source_session"],
                        trace=str(path),
                        trace_sha256=hashlib.sha256(raw).hexdigest(),
                        turn=turn["turn"],
                        recorded_input_tokens=turn["tokens_in"],
                        mode=mode,
                        body=turn["call_kwargs"],
                    )
                )
    results = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        futures = {
            pool.submit(replay, args.url, record, args.output, args.max_tokens): record
            for record in records
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(dict(id=futures[future]["id"], error=repr(error)))
    summary = dict(
        requests=len(records), passed=len(results), errors=errors, results=results
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if errors:
        raise RuntimeError(f"{len(errors)} SWE trace requests failed: {errors}")
    print(f"PASS: {len(results)} recorded SWE-bench requests", flush=True)


if __name__ == "__main__":
    main()
