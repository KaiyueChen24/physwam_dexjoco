#!/usr/bin/env python3
"""Select paired EgoDex test files from Apple's ZIP using HTTP byte ranges.

Python standard library only. Without --execute, just print the download plan.
Source: https://github.com/apple-aiml-research/ml-egodex
"""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path, PurePosixPath
import random
import re
import time
import urllib.error
import urllib.request
import zipfile
import zlib

URL = "https://ml-site.cdn-apple.com/datasets/egodex/test.zip"
DEFAULT_TASKS = (
    "basic_pick_place", "pour", "screw_unscrew_bottle_cap", "insert_remove_usb",
    "fold_unfold_paper_basic", "tie_and_untie_shoelace",
    "assemble_disassemble_legos", "make_sandwich",
)


class RangeFile(io.RawIOBase):
    """Seekable HTTP object; reject servers/proxies that ignore byte ranges."""

    def __init__(self, url):
        super().__init__()
        self.url, self.position = url, 0
        self.size, self.etag = None, None
        self.cache_start, self.cache = 0, b""
        self.transferred_bytes = 0
        self._fetch(0, 0)

    def _fetch(self, start, end):
        for attempt in range(4):
            headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
            if self.etag and not self.etag.startswith("W/"):
                headers["If-Match"] = self.etag
            request = urllib.request.Request(self.url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if response.status != 206 or not match:
                        raise ValueError("Server/proxy ignored HTTP Range; refusing a full ZIP download")
                    first, last, total = map(int, match.groups())
                    if (first, last) != (start, end) or (self.size is not None and total != self.size):
                        raise ValueError(f"Wrong/stale HTTP range: {content_range}")
                    etag = response.headers.get("ETag")
                    if self.etag is not None and etag != self.etag:
                        raise ValueError("Apple ZIP identity changed during download; rerun the command")
                    payload = response.read(end - start + 2)
                    if len(payload) != end - start + 1:
                        raise OSError("Incomplete HTTP range response")
                    self.size, self.etag = total, etag
                    self.transferred_bytes += len(payload)
                    return payload
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                    raise
                if attempt == 3:
                    raise
                print(f"Network retry {attempt + 1}/3: {type(error).__name__}", flush=True)
                time.sleep(2 ** attempt)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            raise ValueError("Invalid seek mode")
        position = offset + (self.position if whence == io.SEEK_CUR else self.size if whence == io.SEEK_END else 0)
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        if size > 8 * 1024 ** 2:
            raise ValueError("Unexpected large ZIP read; refusing an unbounded download")
        start = self.position
        offset = start - self.cache_start
        if offset < 0 or offset + size > len(self.cache):
            end = min(self.size, start + max(size, 65536)) - 1
            self.cache = self._fetch(start, end)
            self.cache_start, offset = start, 0
        result = self.cache[offset:offset + size]
        self.position += len(result)
        return result


def select_files(archive, tasks, per_task, seed):
    if per_task < 1 or not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Choose distinct tasks and a positive --per-task")
    entries = archive.infolist()
    if len({entry.filename for entry in entries}) != len(entries):
        raise ValueError("Duplicate ZIP member names")
    by_name = {entry.filename: entry for entry in entries}
    rng, selected = random.Random(seed), []
    for task in tasks:
        if not re.fullmatch(r"[a-z0-9_]+", task):
            raise ValueError(f"Invalid task: {task}")
        candidates = [entry for entry in entries if
                      PurePosixPath(entry.filename).parent.as_posix() == f"test/{task}"
                      and PurePosixPath(entry.filename).suffix == ".mp4"
                      and PurePosixPath(entry.filename).stem.isdigit()]
        candidates.sort(key=lambda entry: int(PurePosixPath(entry.filename).stem))
        if len(candidates) < per_task:
            raise ValueError(f"{task}: requested {per_task} videos, found {len(candidates)}")
        videos = sorted(rng.sample(candidates, per_task), key=lambda entry: int(PurePosixPath(entry.filename).stem))
        for video in videos:
            hdf5 = str(PurePosixPath(video.filename).with_suffix(".hdf5"))
            if hdf5 not in by_name:
                raise ValueError(f"Missing paired HDF5: {hdf5}")
            selected.extend((video, by_name[hdf5]))
    return selected


def valid_file(path, entry):
    if not path.is_file() or path.stat().st_size != entry.file_size:
        return False
    checksum = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum == entry.CRC


def download_file(archive, entry, output):
    root = Path(output).resolve()
    target = root.joinpath(*PurePosixPath(entry.filename).parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError("ZIP member escapes the output directory")
    if target.exists():
        if valid_file(target, entry):
            print(f"Verified existing: {entry.filename}", flush=True)
            return
        raise ValueError(f"Existing file has wrong size/CRC; inspect it before replacing: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    if temporary.is_symlink():
        raise ValueError(f"Refusing a symlink temporary file: {temporary}")
    try:
        # ZipExtFile verifies decompressed CRC at EOF, including deflated members.
        with archive.open(entry) as incoming, temporary.open("wb") as outgoing:
            for chunk in iter(lambda: incoming.read(4 * 1024 ** 2), b""):
                outgoing.write(chunk)
        if not valid_file(temporary, entry):
            raise ValueError(f"Downloaded ZIP member failed CRC: {entry.filename}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Dataset parent; files go under OUTPUT/test/TASK/")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--per-task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute", action="store_true", help="Download the selected files; default only previews")
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    try:
        with RangeFile(URL) as remote, zipfile.ZipFile(remote) as archive:
            selected = select_files(archive, args.tasks, args.per_task, args.seed)
            counts = Counter(PurePosixPath(entry.filename).parent.name for entry in selected if entry.filename.endswith(".mp4"))
            plan = {"schema_version": 1, "source_url": URL, "archive_etag": remote.etag,
                    "archive_bytes": remote.size, "seed": args.seed, "episodes_per_task": dict(counts),
                    "compressed_bytes": sum(entry.compress_size for entry in selected),
                    "extracted_bytes": sum(entry.file_size for entry in selected),
                    "files": [{"path": entry.filename, "bytes": entry.file_size,
                               "compressed_bytes": entry.compress_size, "crc32": f"{entry.CRC:08x}"} for entry in selected]}
            for task, count in counts.items():
                print(f"{task}: {count} episodes", flush=True)
            print(f"Selected payload: {plan['compressed_bytes'] / 1024**2:.2f} MiB; "
                  f"extracted: {plan['extracted_bytes'] / 1024**2:.2f} MiB", flush=True)
            if not args.execute:
                print("Preview only: add --execute to download. No output files written.", flush=True)
                return 0
            output.mkdir(parents=True, exist_ok=True)
            for number, entry in enumerate(selected, 1):
                print(f"[{number}/{len(selected)}] {entry.filename} ({entry.file_size / 1024**2:.2f} MiB)", flush=True)
                download_file(archive, entry, output)
            # Only a fully verified selection receives a completed manifest.
            temporary = output / "download_manifest.json.part"
            temporary.write_text(json.dumps(plan, indent=2) + "\n")
            temporary.replace(output / "download_manifest.json")
            print(f"Complete: {sum(counts.values())} episodes; {output / 'download_manifest.json'}", flush=True)
            print(f"HTTP payload read this run: {remote.transferred_bytes / 1024**2:.2f} MiB", flush=True)
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"EgoDex download failed: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
