"""Verify a source-only package's ZIP integrity and embedded SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def verify(path: str) -> int:
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"CRC failure: {corrupt}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate member names")
        if any("__pycache__" in name or name.endswith(".pyc") for name in names):
            raise ValueError("ZIP contains Python cache files")
        if any(name.startswith(("data/", "results/")) for name in names):
            raise ValueError("source-only ZIP contains data or results")

        manifest = json.loads(archive.read("PUBLIC_PACKAGE_MANIFEST.json"))
        listed = {item["path"]: item for item in manifest["files"]}
        actual = set(names) - {"PUBLIC_PACKAGE_MANIFEST.json"}
        if actual != set(listed):
            missing = sorted(set(listed) - actual)
            unlisted = sorted(actual - set(listed))
            raise ValueError(f"Manifest membership mismatch; missing={missing}; unlisted={unlisted}")

        for name, item in listed.items():
            content = archive.read(name)
            if len(content) != item["bytes"]:
                raise ValueError(f"Byte-length mismatch: {name}")
            # Hex digests are case-insensitive; manifests created by external
            # tools may use lowercase while this verifier formats uppercase.
            if sha256_bytes(content).casefold() != item["sha256"].casefold():
                raise ValueError(f"SHA-256 mismatch: {name}")

    return len(listed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path")
    args = parser.parse_args()
    count = verify(args.zip_path)
    print(f"PASS: ZIP CRC, package boundaries, and {count} manifest entries verified")


if __name__ == "__main__":
    main()
