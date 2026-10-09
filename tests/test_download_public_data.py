"""Offline tests for source-data download integrity and safe extraction."""

from __future__ import annotations

import hashlib
import io
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src import download_public_data as downloader


class PublicDataDownloadTests(unittest.TestCase):
    def test_all_continues_with_zenodo_when_osf_denies_anonymous_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = io.StringIO()
            with (
                patch.object(sys, "argv", ["download_public_data.py", "--dataset", "all", "--raw-root", temporary]),
                patch.object(downloader, "download_dishbrain", side_effect=PermissionError("OSF HTTP 401")) as dishbrain,
                patch.object(downloader, "download_organoid", return_value=[("MAT", "downloaded and verified")]) as organoid,
                redirect_stdout(output),
            ):
                with self.assertRaises(SystemExit) as raised:
                    downloader.main()
        self.assertEqual(raised.exception.code, 1)
        dishbrain.assert_called_once()
        organoid.assert_called_once()
        self.assertIn("downloaded and verified: MAT", output.getvalue())
        self.assertIn("FAILED to retrieve dishbrain: OSF HTTP 401", output.getvalue())

    def test_osf_auth_error_explains_offline_hash_verification_fallback(self):
        error = urllib.error.HTTPError("https://api.osf.io/v2/nodes/xwf7v/", 401, "Unauthorized", {}, None)
        with patch.object(downloader, "_get_json", side_effect=error):
            with self.assertRaises(PermissionError) as raised:
                downloader._list_osf_files()
        self.assertIn("No credentials were sent", str(raised.exception))
        self.assertIn("--verify-existing", str(raised.exception))

    def test_verify_existing_hash_checks_local_dishbrain_files_without_network(self):
        payload = b"audited local fixture"
        specification = {
            "sample.bin": {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        }
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary)
            target = raw_root / "dishbrain" / "sample.bin"
            target.parent.mkdir()
            target.write_bytes(payload)
            with patch.dict(downloader.DISHBRAIN_FILES, specification, clear=True):
                with patch.object(downloader, "_open_url") as open_url:
                    reports = downloader.verify_dishbrain(raw_root)
            open_url.assert_not_called()
        self.assertEqual(reports, [("sample.bin", "local copy verified; no network request")])

    def test_verify_existing_rejects_mismatched_local_file(self):
        payload = b"wrong local bytes"
        specification = {"sample.bin": {"size": 8, "sha256": hashlib.sha256(b"expected").hexdigest()}}
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary)
            target = raw_root / "dishbrain" / "sample.bin"
            target.parent.mkdir()
            target.write_bytes(payload)
            with patch.dict(downloader.DISHBRAIN_FILES, specification, clear=True):
                with self.assertRaises(ValueError):
                    downloader.verify_dishbrain(raw_root)
            self.assertEqual(target.read_bytes(), payload)

    def test_download_promotes_only_hash_verified_content(self):
        payload = b"audited bytes"
        expected_hash = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "source.bin"
            with patch.object(downloader, "_open_url") as open_url:
                from io import BytesIO

                open_url.return_value.__enter__.return_value = BytesIO(payload)
                status = downloader._download_verified("https://example.org/file", target, len(payload), expected_hash)
            self.assertEqual(status, "downloaded and verified")
            self.assertEqual(target.read_bytes(), payload)

    def test_download_rejects_wrong_bytes_without_leaving_target(self):
        payload = b"wrong bytes"
        expected_hash = hashlib.sha256(b"expected").hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "source.bin"
            with patch.object(downloader, "_open_url") as open_url:
                from io import BytesIO

                open_url.return_value.__enter__.return_value = BytesIO(payload)
                with self.assertRaises(ValueError):
                    downloader._download_verified("https://example.org/file", target, 8, expected_hash)
            self.assertFalse(target.exists())

    def test_extracts_only_mat_files_into_flat_safe_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "source.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for index in range(45):
                    archive.writestr(f"nested/record_{index:02d}.mat", b"mat")
                archive.writestr("README.txt", b"not extracted")
            output = root / "Indiv_organoids"
            status = downloader._extract_mat_files(archive_path, output)
            self.assertEqual(status, "extracted 45 MAT files")
            self.assertEqual(len(list(output.glob("*.mat"))), 45)
            self.assertFalse((output / "README.txt").exists())

    def test_rejects_path_traversal_in_mat_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for index in range(44):
                    archive.writestr(f"record_{index:02d}.mat", b"mat")
                archive.writestr("../escape.mat", b"unsafe")
            with self.assertRaises(ValueError):
                downloader._extract_mat_files(archive_path, root / "out")
            self.assertFalse((root / "escape.mat").exists())


if __name__ == "__main__":
    unittest.main()
