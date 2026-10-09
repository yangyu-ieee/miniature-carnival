"""Download the exact public source files used by NeuroChip Copilot.

This script retrieves data from the original OSF and Zenodo APIs; it does not
re-host or bundle source data. Check each source's current terms before use.
In particular, the OSF project's file-level redistribution license is not
confirmed, so this script only downloads into the user's local data directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "NeuroChip-Copilot-data-reproducer/1.0"
CHUNK_BYTES = 1024 * 1024

# These values are pinned to the files audited for this project. The checksums
# identify bytes; they do not grant a license or redistribution permission.
DISHBRAIN_FILES = {
    "drug_game_stats": {
        "size": 69317,
        "sha256": "bb9aa84c010337bcab0f7e66ceb2356f39b26c9f2bf024e5480fd5b757025f3a",
    },
    "drug_testing_firing_stats.feather": {
        "size": 38306,
        "sha256": "3a6e90a810afa3a9f7b123940af575fde9f016b98925016ffea20caa76e9f67e",
    },
    "criticality_metrics_drug": {
        "size": 91874,
        "sha256": "6cac57d32ffe4373a7d9f4153df4a1e0d3b062a23adb8994145a9f13856266b6",
    },
    "burst_pattern_class_drug": {
        "size": 67737,
        "sha256": "a1e56d86ac80567cee2fd9aaedcb3a24bba1a313dbef6cdfe9433b7af8619425",
    },
}

ORGANOID_FILES = {
    "all_fig_table_data_revision2.xlsx": {
        "record": "22755004",
        "size": 317996,
        "sha256": "aa7181af9b352915dded058687489f49f572c125ab3267c65fe9173df19b7fef",
        "target": "all_fig_table_data_revision2.xlsx",
    },
    "Indiv_organoids.zip": {
        "record": "22755004",
        "size": 7606972,
        "sha256": "49976511c5240091ad5af9390ff6c131b56e6c20ea128affb9f93cf28bbd6ad5",
        "target": "Indiv_organoids.zip",
    },
    "fig_table_source_data.xlsx": {
        "record": "20286251",
        "size": 265298,
        "sha256": "03e7c0bb55ffd377787ced84ce6317a81e1f5d567d18653122c779e8b027d02f",
        "target": "fig_table_source_data.xlsx",
    },
}


def _open_url(url: str, timeout: int = 60):
    """Open an HTTPS source with a reproducible user-agent and finite timeout."""
    if not url.startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS source URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def _get_json(url: str) -> dict[str, Any]:
    """Read one source API page without adding a third-party HTTP dependency."""
    with _open_url(url) as response:
        return json.loads(response.read().decode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so large source archives need little memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified(url: str, destination: Path, expected_size: int, expected_sha256: str) -> str:
    """Download to a sibling temporary file; promote only after both checks pass."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = sha256_file(destination)
        if destination.stat().st_size == expected_size and actual == expected_sha256:
            return "already verified"
        raise FileExistsError(
            f"Existing file differs from pinned source; left untouched: {destination} "
            f"(size={destination.stat().st_size}, sha256={actual})"
        )

    digest = hashlib.sha256()
    size = 0
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output, _open_url(url) as response:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        actual_hash = digest.hexdigest()
        if size != expected_size or actual_hash != expected_sha256:
            raise ValueError(
                f"Downloaded bytes do not match the audited source for {destination.name}: "
                f"size {size}/{expected_size}, sha256 {actual_hash}/{expected_sha256}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "downloaded and verified"


def _list_osf_files() -> dict[str, dict[str, Any]]:
    """Try to enumerate OSF files without credentials; access may be restricted."""
    url = "https://api.osf.io/v2/nodes/xwf7v/files/osfstorage/?page%5Bsize%5D=100"
    found: dict[str, dict[str, Any]] = {}
    try:
        while url:
            page = _get_json(url)
            for item in page.get("data", []):
                attributes = item.get("attributes", {})
                if attributes.get("kind") != "file":
                    continue
                name = attributes.get("name")
                download = item.get("links", {}).get("download")
                if name and download:
                    found[name] = {"download": download, "size": attributes.get("size")}
            url = page.get("links", {}).get("next")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise PermissionError(
                f"OSF returned HTTP {exc.code} for anonymous access to project xwf7v. "
                "No credentials were sent. If you have authorized access, obtain the four "
                "files through the source's permitted route, place them under "
                "data/raw/dishbrain/, then run this script with --dataset dishbrain "
                "--verify-existing to check their pinned sizes and hashes."
            ) from exc
        raise
    return found


def download_dishbrain(raw_root: Path) -> list[tuple[str, str]]:
    """Fetch and verify the four pinned OSF files; never copy them into a repo."""
    remote = _list_osf_files()
    missing = sorted(set(DISHBRAIN_FILES) - set(remote))
    if missing:
        raise FileNotFoundError(f"Pinned filenames were not found in the current OSF listing: {missing}")

    reports = []
    for name, expected in DISHBRAIN_FILES.items():
        resource = remote[name]
        if resource["size"] is not None and int(resource["size"]) != expected["size"]:
            raise ValueError(f"OSF metadata size changed for {name}: {resource['size']} != {expected['size']}")
        status = _download_verified(
            resource["download"], raw_root / "dishbrain" / name, expected["size"], expected["sha256"]
        )
        reports.append((name, status))
    return reports


def verify_dishbrain(raw_root: Path) -> list[tuple[str, str]]:
    """Verify authorized local copies without making a network request."""
    reports = []
    for name, expected in DISHBRAIN_FILES.items():
        path = raw_root / "dishbrain" / name
        if not path.is_file():
            raise FileNotFoundError(f"Required local source file is missing: {path}")
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != expected["size"] or actual_hash != expected["sha256"]:
            raise ValueError(
                f"Local source file does not match the audited bytes for {name}: "
                f"size {actual_size}/{expected['size']}, sha256 {actual_hash}/{expected['sha256']}"
            )
        reports.append((name, "local copy verified; no network request"))
    return reports


def _zenodo_file(record_id: str, filename: str) -> dict[str, Any]:
    """Resolve a named file from the original Zenodo record API."""
    record = _get_json(f"https://zenodo.org/api/records/{record_id}")
    for item in record.get("files", []):
        if item.get("key") == filename:
            links = item.get("links", {})
            return {"download": links.get("download") or links.get("self"), "size": item.get("size")}
    raise FileNotFoundError(f"{filename} was not listed in Zenodo record {record_id}")


def _extract_mat_files(archive_path: Path, output_dir: Path, expected_count: int = 45) -> str:
    """Safely extract only MAT basenames from the audited archive, rejecting path traversal."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="organoid-mat-stage-", dir=output_dir.parent) as stage_name:
        stage = Path(stage_name)
        with zipfile.ZipFile(archive_path) as archive:
            members = [item for item in archive.infolist() if item.filename.lower().endswith(".mat")]
            if len(members) != expected_count:
                raise ValueError(f"Expected {expected_count} MAT files in source archive; found {len(members)}")
            seen: set[str] = set()
            for member in members:
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or ".." in relative.parts or member.is_dir():
                    raise ValueError(f"Unsafe MAT archive member path: {member.filename}")
                name = relative.name
                if name in seen:
                    raise ValueError(f"Duplicate MAT basename in archive: {name}")
                seen.add(name)
                target = stage / name
                with archive.open(member) as source, target.open("xb") as output:
                    while chunk := source.read(CHUNK_BYTES):
                        output.write(chunk)
        if output_dir.exists():
            existing = {path.name: path for path in output_dir.glob("*.mat")} if output_dir.is_dir() else {}
            staged = {path.name: path for path in stage.glob("*.mat")}
            if set(existing) != set(staged):
                raise FileExistsError(f"Existing MAT directory differs from source archive; left untouched: {output_dir}")
            for name, expected_file in staged.items():
                if sha256_file(existing[name]) != sha256_file(expected_file):
                    raise FileExistsError(f"Existing MAT file differs from source archive; left untouched: {existing[name]}")
            return "already present and verified against source archive"
        stage.rename(output_dir)
    return f"extracted {expected_count} MAT files"


def download_organoid(raw_root: Path) -> list[tuple[str, str]]:
    """Fetch Zenodo v3 files plus the v2 comparison table and verify local hashes."""
    reports = []
    for filename, spec in ORGANOID_FILES.items():
        resource = _zenodo_file(spec["record"], filename)
        if int(resource["size"]) != spec["size"]:
            raise ValueError(f"Zenodo metadata size changed for {filename}: {resource['size']} != {spec['size']}")
        target = raw_root / spec["target"]
        status = _download_verified(resource["download"], target, spec["size"], spec["sha256"])
        reports.append((filename, status))
    extract_status = _extract_mat_files(raw_root / "Indiv_organoids.zip", raw_root / "Indiv_organoids")
    reports.append(("Indiv_organoids/*.mat", extract_status))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "dishbrain", "organoid"), default="all")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="verify the four locally obtained DishBrain files by size/SHA-256 without network access",
    )
    args = parser.parse_args()

    if args.verify_existing and args.dataset != "dishbrain":
        parser.error("--verify-existing is available only with --dataset dishbrain")

    print("Source terms may change. This script downloads original files to your local directory only.")
    print("It does not bundle or re-host data; checksums verify bytes, not reuse or redistribution rights.")
    reports: list[tuple[str, str]] = []
    failures: list[tuple[str, Exception]] = []
    if args.verify_existing:
        reports.extend(verify_dishbrain(args.raw_root))
    elif args.dataset in ("all", "dishbrain"):
        try:
            reports.extend(download_dishbrain(args.raw_root))
        except Exception as exc:
            failures.append(("dishbrain", exc))
    if args.dataset in ("all", "organoid"):
        try:
            reports.extend(download_organoid(args.raw_root))
        except Exception as exc:
            failures.append(("organoid", exc))
    for name, status in reports:
        print(f"{status}: {name}")
    for dataset, error in failures:
        print(f"FAILED to retrieve {dataset}: {error}")
    if failures:
        print("Some selected datasets may have been downloaded and verified; inspect the per-file lines above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
