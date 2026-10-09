"""Build a deterministic file/hash manifest for a source-only review package.

This helper intentionally excludes the manifest itself to avoid a self-hash
cycle. Run it on the staging directory immediately before making the ZIP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_NAME = "PUBLIC_PACKAGE_MANIFEST.json"
IGNORED_DIRECTORY_NAMES = {".git", ".mplconfig", ".pytest_cache", ".venv", "__pycache__", "venv"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_manifest(
    root: Path,
    version: str,
    base_archive: str,
    base_archive_sha256: str,
    changes: list[str] | None = None,
    analysis_added: bool = True,
    status: str = "INTERNAL_ONLY_NOT_APPROVED_FOR_PUBLICATION_OR_SUBMISSION",
) -> dict:
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if any(part in IGNORED_DIRECTORY_NAMES for part in path.relative_to(root).parts):
            continue
        relative = path.relative_to(root).as_posix()
        if relative == MANIFEST_NAME:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "version": version,
        "status": status,
        "base_archive": base_archive,
        "base_archive_sha256": base_archive_sha256.upper(),
        "changes": changes or [
            "README.md",
            "DATASET_BASELINE_ROW_SELECTION_CORRECTION_20261003.md",
            "INTERNAL_SUBMISSION_CANDIDATE_STATUS.md",
            "app.py (corrected project lineage and cache version bump)",
            "src/audit_dishbrain.py",
            "tests/test_earliest_rest_semantics.py",
            "tests/test_evidence_contract.py",
            "tests/test_redteam_dashboard_ui.py (corrected lineage and cache regression checks)",
            "corrected figures",
        ],
        "analysis_added": analysis_added,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="package staging directory")
    parser.add_argument("--version", required=True)
    parser.add_argument("--base-archive", required=True)
    parser.add_argument("--base-archive-sha256", required=True)
    parser.add_argument(
        "--change",
        action="append",
        dest="changes",
        help="Describe one change in this package; may be supplied multiple times.",
    )
    parser.add_argument(
        "--analysis-added",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether this revision adds scientific analysis (default: true).",
    )
    parser.add_argument(
        "--status",
        default="INTERNAL_ONLY_NOT_APPROVED_FOR_PUBLICATION_OR_SUBMISSION",
        help="Release status recorded in the manifest.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"package root does not exist: {root}")
    manifest = build_manifest(
        root,
        args.version,
        args.base_archive,
        args.base_archive_sha256,
        changes=args.changes,
        analysis_added=args.analysis_added,
        status=args.status,
    )
    target = root / MANIFEST_NAME
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {target} with {len(manifest['files'])} file entries")


if __name__ == "__main__":
    main()
