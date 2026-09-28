"""Compare the training reward with Gym's raw reward per sample to detect reward penalties (reward -> 0)."""
import glob
import json
import os
import sys
from collections import Counter

RUN = sys.argv[1]
MAX_JL = int(os.environ.get("MAX_JL", "9999"))
DUMPS = os.environ.get("DUMPS_DIR") or os.path.join(RUN, "dumps")
pairs = Counter()
n = 0
flags_when_zeroed = Counter()
for jl in sorted(glob.glob(os.path.join(DUMPS, "rollouts/*.jsonl")))[:MAX_JL]:
    with open(jl) as f:
        for line in f:
            rec = json.loads(line)
            r = rec.get("reward")
            fr = (rec.get("full_result") or {}).get("reward")
            pairs[(None if r is None else round(float(r), 3), None if fr is None else round(float(fr), 3))] += 1
            n += 1
            if fr is not None and r is not None and float(fr) > 0 and float(r) == 0:
                msgs = rec["messages"]
                flags_when_zeroed["invalid_tool_call"] += int(any(m.get("is_invalid_tool_call") for m in msgs))
                flags_when_zeroed["malformed_thinking"] += int(any(m.get("has_malformed_thinking") for m in msgs))
                flags_when_zeroed["n"] += 1
print(f"{os.path.basename(RUN)}: {n} records; (training reward, gym reward) pairs: {sorted(pairs.items(), key=lambda kv: -kv[1])[:8]}")
print(f"  gym reward > 0 but training reward == 0: {flags_when_zeroed['n']} (with invalid_tool_call flag {flags_when_zeroed['invalid_tool_call']}, malformed_thinking flag {flags_when_zeroed['malformed_thinking']})")
