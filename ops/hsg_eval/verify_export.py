import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
    assert (root / name).is_file() and (root / name).stat().st_size > 0, name
    json.loads((root / name).read_text())
cfg = json.loads((root / "config.json").read_text())
assert "NemotronHForCausalLM" in cfg.get("architectures", []), cfg.get("architectures")
index = json.loads((root / "model.safetensors.index.json").read_text())
expected = set(index["weight_map"].values())
actual = {p.name for p in root.glob("model*.safetensors")}
assert len(expected) == 14 and actual == expected, (len(expected), actual ^ expected)
for name in expected:
    p = root / name
    assert p.stat().st_size > 0, name
    with open(p, "rb") as f:
        assert len(f.read(8)) == 8, f"unreadable {name}"
print(json.dumps({"hf_export": str(root), "shards": len(actual),
                  "bytes": sum((root / name).stat().st_size for name in actual),
                  "index_tensor_count": len(index["weight_map"]), "status": "verified"}))
