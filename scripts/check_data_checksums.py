#!/usr/bin/env python
"""Verify that the dataset files match the committed checksums.

Reads `data/checksums.txt` (which is `sha256sum data/**/*.json` output) and
checks every listed file. Exits with rc=1 on any mismatch.
"""

import hashlib
import sys
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    repo_root = Path(__file__).resolve().parents[1]
    checksums = repo_root / "data" / "checksums.txt"
    if not checksums.exists():
        print(f"FAIL: {checksums} is missing.")
        return 1

    fail = 0
    with open(checksums) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, _, rel = line.partition("  ")
            target = repo_root / rel
            if not target.exists():
                print(f"MISSING: {rel}")
                fail += 1
                continue
            actual = sha256_file(target)
            if actual != digest:
                print(f"MISMATCH: {rel}\n  expected: {digest}\n  actual:   {actual}")
                fail += 1

    if fail:
        print(f"\n{fail} file(s) failed integrity check.")
        return 1
    print("All checksums OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
