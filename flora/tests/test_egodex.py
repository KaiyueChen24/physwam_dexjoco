"""EgoDex manifest selection, blank-caption isolation and timing accounting."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zlib

from physwam_flora.common import read_manifest, validate_manifest, write_json
from physwam_flora.datasets import prepare
from physwam_flora.evaluation import caption_timing
from physwam_flora.jobs import build_jobs
from physwam_flora.runner import run

HAS_MEDIA = all(importlib.util.find_spec(name) for name in ("av", "numpy", "PIL"))


@unittest.skipUnless(HAS_MEDIA, "PyAV/numpy/Pillow required")
class EgoDex(unittest.TestCase):
    def fixture(self, root):
        from test_pipeline import video

        seed = root / "source.mp4"
        video(seed, frames=30)
        dataset = root / "dataset"
        entries = []
        for task in ("pour", "insert_remove_usb"):
            for index in (2, 7, 9):
                for extension in ("mp4", "hdf5"):
                    name = f"test/{task}/{index}.{extension}"
                    path = dataset / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # HDF5 is deliberately opaque to this adapter; no text/poses enter the model.
                    data = seed.read_bytes() if extension == "mp4" else b"opaque reference metadata fixture"
                    path.write_bytes(data)
                    entries.append({"path": name, "bytes": len(data), "crc32": f"{zlib.crc32(data):08x}"})
        write_json(dataset / "download_manifest.json", {"schema_version": 1, "files": entries})
        return {"schema_version": 1, "source": {"kind": "egodex", "root": str(dataset), "limit_per_task": 3},
                "tasks": None, "caption": {"profile": "human", "views": "main"},
                "runtime": {"captioner_python": sys.executable}}

    def test_selected_files_only_and_official_human_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            # A leftover MP4 not in the completed download selection must not be included.
            extra = root / "dataset/test/pour/999.mp4"
            extra.write_bytes(b"not selected")
            result = prepare(config, root / "input")
            rows = read_manifest(result["manifest"])
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["episode_index"] for row in rows}, {2, 7, 9})
            self.assertTrue(all(row["instruction"] == "" and row["success"] is None for row in rows))
            jobs, _ = build_jobs(rows, config, "caption")
            self.assertTrue(all(job["profile"] == "human" and job["instruction"] == "" for job in jobs))
            self.assertTrue(all(set(job) == {"id", "video", "profile", "instruction", "yam32"} for job in jobs))
            self.assertNotIn("reference", json.dumps(jobs))
            with self.assertRaisesRegex(ValueError, "caption-only"):
                build_jobs(rows, config, "terminal")
            run(config, result["manifest"], root / "validated", "caption", native_check=True)
            output = (root / "validated/run.log").read_text()
            self.assertEqual(output.count('"new_model_calls": 0'), 6)
            self.assertIn('"profile": "Human"', output)

    def test_crc_and_short_task_fail_instead_of_silent_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            config["source"]["limit_per_task"] = 4
            with self.assertRaisesRegex(ValueError, "requested 4 episodes"):
                prepare(config, root / "short")
            config["source"]["limit_per_task"] = 3
            hdf5 = root / "dataset/test/pour/2.hdf5"
            hdf5.write_bytes(b"x" * hdf5.stat().st_size)
            with self.assertRaisesRegex(ValueError, "CRC mismatch"):
                prepare(config, root / "corrupt")

    def test_blank_mode_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existence-only.mp4"
            path.touch()
            row = {"id": "fixture", "instruction": "", "cameras": {"base": str(path)}}
            with self.assertRaises(ValueError):
                validate_manifest([row])
            row["instruction_mode"] = "none"
            validate_manifest([row])
            row["instruction"] = "reference text"
            with self.assertRaisesRegex(ValueError, "empty instruction"):
                validate_manifest([row])


class Timing(unittest.TestCase):
    def test_dual_mean_uses_only_dual_and_requires_complete_timings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "status.json", {"status": "complete"})
            write_json(root / "run.json", {"plan": {"stage": "caption", "mode": "inference"}, "episodes": [
                {"id": "single", "task_name": "pick_bucket", "robot": "single_arm"},
                {"id": "dual_a", "task_name": "bimanual_assembly", "robot": "dual_arm"},
                {"id": "dual_b", "task_name": "bimanual_assembly", "robot": "dual_arm"}]})
            for name, elapsed in (("single", 10), ("dual_a", 100), ("dual_b", 300)):
                write_json(root / f"annotations/{name}.json", {"elapsed_s": elapsed, "duration_s": 20, "segments": [{}, {}]})
            stats = caption_timing(root, "dual")
            self.assertEqual(stats["episodes"], 2)
            self.assertEqual(stats["mean_s"], 200)
            self.assertEqual(stats["processing_s_per_video_s"], 10)
            self.assertEqual(stats["by_task"]["bimanual_assembly"]["mean_s"], 200)
            path = root / "annotations/dual_b.json"
            path.write_text(json.dumps({"duration_s": 20, "segments": []}))
            with self.assertRaisesRegex(ValueError, "refusing a partial average"):
                caption_timing(root, "dual")


if __name__ == "__main__":
    unittest.main()
