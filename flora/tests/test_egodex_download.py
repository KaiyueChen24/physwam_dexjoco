"""Exercise remote ZIP/ZIP64 extraction without real network access."""
import importlib.util
import io
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_egodex.py"
SPEC = importlib.util.spec_from_file_location("egodex_download", SCRIPT)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class Response(io.BytesIO):
    def __init__(self, data, status, headers):
        super().__init__(data)
        self.status, self.headers = status, headers


class RemoteZip(unittest.TestCase):
    def fixture(self):
        buffer = io.BytesIO()
        # Exercise ZIP64 end records even though these fixtures are small.
        with patch.object(zipfile, "ZIP64_LIMIT", 64):
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for task in download.DEFAULT_TASKS:
                    for index in range(5):
                        for suffix in (".mp4", ".hdf5"):
                            archive.writestr(f"test/{task}/{index}{suffix}", f"fixture {task} {index} {suffix}" * 100)
        return buffer.getvalue()

    def responder(self, data):
        def open_range(request, timeout):
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
            return Response(data[start:end + 1], 206,
                            {"Content-Range": f"bytes {start}-{end}/{len(data)}", "ETag": '"fixture"'})
        return open_range

    def test_zip64_selection_extract_and_verified_restart(self):
        data = self.fixture()
        with patch.object(download.urllib.request, "urlopen", side_effect=self.responder(data)):
            with download.RangeFile(download.URL) as remote, zipfile.ZipFile(remote) as archive:
                selected = download.select_files(archive, list(download.DEFAULT_TASKS), 3, 0)
                self.assertEqual(len(selected), 48)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    for entry in selected:
                        download.download_file(archive, entry, root)
                        self.assertTrue(download.valid_file(root / entry.filename, entry))
                    before = remote.transferred_bytes
                    download.download_file(archive, selected[0], root)
                    self.assertEqual(remote.transferred_bytes, before)
                    path = root / selected[0].filename
                    path.write_bytes(b"corrupt data")
                    with self.assertRaisesRegex(ValueError, "wrong size/CRC"):
                        download.download_file(archive, selected[0], root)
                    self.assertEqual(path.read_bytes(), b"corrupt data")

    def test_reject_full_download_and_stale_proxy_range(self):
        for status, headers in ((200, {}), (206, {"Content-Range": "bytes 5-5/100"})):
            response = Response(b"should never read", status, headers)
            with patch.object(download.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    download.RangeFile(download.URL)

    def test_exact_counts_and_no_path_escape(self):
        with zipfile.ZipFile(io.BytesIO(self.fixture())) as archive:
            for tasks, count in ((["missing_task"], 3), (["pour"], 6), (["pour", "pour"], 1)):
                with self.assertRaises(ValueError):
                    download.select_files(archive, tasks, count, 0)
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "escapes"):
                    download.download_file(archive, zipfile.ZipInfo("../outside"), directory)


if __name__ == "__main__":
    unittest.main()
