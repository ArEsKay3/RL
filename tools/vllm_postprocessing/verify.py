# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Verify vendored output hashes and optionally fetch/check pinned upstream files."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "nemo_rl/_vendor/vllm"
    manifest = json.loads((root / "manifest.json").read_text())
    for name, entry in manifest["generated"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"Vendored file changed: {name}")
    if args.upstream:
        for name, entry in manifest["files"].items():
            with urlopen(entry["url"], timeout=30) as response:
                actual = hashlib.sha256(response.read()).hexdigest()
            if actual != entry["sha256"]:
                raise ValueError(f"Upstream source mismatch: {name}")
    print(
        f"Verified {len(manifest['generated'])} generated files"
        + (f" and {len(manifest['files'])} upstream files" if args.upstream else "")
    )


if __name__ == "__main__":
    main()
