"""CPU integration tests with small real videos; no weight downloads or CUDA."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from physwam_flora.common import VENDOR, load_config, read_manifest, verify_vendor, write_json, write_jsonl
from physwam_flora.datasets import prepare, rollout_rows
from physwam_flora.evaluation import terminal_metrics, dense_quality
from physwam_flora.jobs import build_jobs, validate_rubric
from physwam_flora.media import extract_episode, inspect_episode, probe
from physwam_flora.runner import make_plan, run, verify_results

HAS_MEDIA = all(importlib.util.find_spec(x) for x in ["av", "numpy", "PIL"])


def video(path, frames=60, fps=30):
    import av
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.options = {"crf": "0", "preset": "fast"}
        for index in range(frames):
            pixels = np.full((48, 64, 3), index * 3 % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "video.mp4"
        self.path.write_bytes(b"existence-only fixture; not used for decoding")
        self.row = {"id": "sample", "task_name": "pick_bucket", "instruction": "Lift the bucket.",
                    "cameras": {"base": str(self.path), "wrist": str(self.path)}, "success": False}

    def test_pinned_sources_and_upstream_manifest(self):
        self.assertGreater(verify_vendor()["verified_files"], 50)
        import hashlib

        manifest = json.loads((VENDOR / "data_engine/batch_utils/CODE_MANIFEST.json").read_text())
        for path, spec in manifest["files"].items():
            self.assertEqual(hashlib.sha256((VENDOR / "data_engine" / path).read_bytes()).hexdigest(), spec["sha256"])

    def test_jobs_never_expose_ground_truth(self):
        jobs, labels = build_jobs([self.row], {}, "terminal")
        self.assertEqual(set(jobs[0]), {"id", "task", "videos"})
        self.assertFalse(labels[0]["success"])
        self.assertEqual(jobs[0]["videos"], [str(self.path), str(self.path)])
        jobs, _ = build_jobs([self.row], {}, "caption")
        self.assertNotIn("left_wrist", jobs[0])
        self.assertNotIn("success", jobs[0])

    def test_three_view_order_and_missing_camera(self):
        row = {**self.row, "cameras": {"base": "top", "wrist_left": "left", "wrist_right": "right"}}
        jobs, _ = build_jobs([row], {}, "dense")
        self.assertEqual(jobs[0]["videos"], ["left", "top", "right"])
        jobs, _ = build_jobs([row], {}, "caption")
        self.assertEqual(jobs[0]["video"], "top")
        self.assertEqual(jobs[0]["left_wrist"], "left")
        with self.assertRaises(ValueError):
            build_jobs([self.row], {"caption": {"views": "three"}}, "caption")

    def test_duplicate_manifest_and_missing_video(self):
        path = self.root / "episodes.jsonl"
        write_jsonl(path, [self.row, self.row])
        with self.assertRaises(ValueError):
            read_manifest(path)
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            read_manifest(path)

    def test_pinned_plan_and_venv_symlink(self):
        manifest = self.root / "episodes.jsonl"
        write_jsonl(manifest, [self.row])
        python = self.root / "venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        config = {"runtime": {"rewarder_python": str(python)}}
        plan, *_ = make_plan(config, manifest, self.root / "out", "terminal")
        self.assertEqual(plan["argv"][0], str(python))
        self.assertIn("--revision", plan["argv"])
        self.assertIn("batch_utils.terminal_reward", plan["argv"])
        self.assertFalse((self.root / "out").exists())
        with self.assertRaises(ValueError):
            make_plan({}, manifest, self.root / "out", "rubric")

    def test_rubric_validation(self):
        valid = [{"action": "Lift", "grading": [{"label": "No", "points": 0}, {"label": "Yes", "points": 1}]}]
        validate_rubric(valid)
        valid[0]["grading"][1]["points"] = 0.5
        with self.assertRaises(ValueError):
            validate_rubric(valid)

    def test_incomplete_backend_results_are_not_complete(self):
        write_jsonl(self.root / "terminal.jsonl", [{"id": "a", "score": 5}])
        with self.assertRaises(ValueError):
            verify_results(self.root, "terminal", ["a", "b"])

    def test_rubric_gpu_budget(self):
        rubric = self.root / "rubric.json"
        write_json(rubric, [{"action": "Lift", "grading": [{"label": "No", "points": 0}, {"label": "Yes", "points": 1}]}])
        manifest = self.root / "episodes.jsonl"
        write_jsonl(manifest, [self.row])
        config = {"rubric": {"files": {"pick_bucket": str(rubric)}}}
        with self.assertRaisesRegex(ValueError, "tensor_parallel_size"):
            make_plan(config, manifest, self.root / "out", "rubric", "0")
        plan, jobs, *_ = make_plan(config, manifest, self.root / "out", "rubric", "0,1")
        self.assertIn("--tensor-parallel-size", plan["argv"])
        self.assertNotIn("success", jobs[0])

    def test_metrics_missing_single_class_and_ties(self):
        labels, predictions = self.root / "labels.jsonl", self.root / "predictions.jsonl"
        write_jsonl(labels, [{"id": "a", "success": True}, {"id": "b", "success": False},
                            {"id": "c", "success": False}, {"id": "d", "success": None}])
        write_jsonl(predictions, [{"id": "a", "score": 5}, {"id": "b", "score": 5}, {"id": "d", "score": 2}])
        metrics = terminal_metrics(predictions, labels)
        self.assertEqual(metrics["auroc"], 0.5)
        self.assertEqual(metrics["failure_recall"], 0)
        self.assertEqual(metrics["missing_predictions"], ["c"])
        self.assertEqual(metrics["unlabelled_episodes"], 1)
        predictions.unlink()
        write_jsonl(predictions, [{"id": "a", "score": 5}])
        self.assertIsNone(terminal_metrics(predictions, labels)["balanced_accuracy"])

    def test_dense_fallback_flag(self):
        path = self.root / "dense.jsonl"
        write_jsonl(path, [{"id": "a", "invalid_clips": 1, "clips": [{}, {}]}])
        self.assertFalse(dense_quality(path)["usable_without_parse_fallback"])


@unittest.skipUnless(HAS_MEDIA, "requires av/numpy/Pillow; install physwam-flora[data]")
class MediaIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.mp4"
        video(self.source)
        self.row = {"id": "sample", "task_name": "pick_bucket", "instruction": "Lift the bucket.",
                    "cameras": {"base": str(self.source)}, "fps": 30, "frames": 60, "success": True}

    def test_native_rate_clip_and_no_frame_leakage(self):
        import av

        target = self.root / "clip.mp4"
        extract_episode(self.source, target, 1.0, 2.0, 30, 30)
        self.assertEqual(probe(target)["frames"], 30)
        with av.open(str(target)) as container:
            first = next(container.decode(video=0)).to_ndarray(format="rgb24")
        self.assertAlmostEqual(float(first.mean()), 90, delta=3)
        with self.assertRaises(ValueError):
            extract_episode(self.source, self.root / "bad.mp4", 1, 2, 30, 29)
        self.assertFalse((self.root / "bad.mp4").exists())

    def test_unsynchronized_videos_rejected(self):
        other = self.root / "short.mp4"
        video(other, frames=59)
        row = {**self.row, "cameras": {"base": str(self.source), "wrist": str(other)}}
        with self.assertRaises(ValueError):
            inspect_episode(row)

    def test_official_caption_cpu_check_and_runner(self):
        manifest = self.root / "episodes.jsonl"
        write_jsonl(manifest, [self.row])
        output = self.root / "run"
        config = {"runtime": {"captioner_python": sys.executable}, "caption": {"profile": "dex", "views": "main"}}
        result = run(config, manifest, output, "caption", native_check=True)
        self.assertEqual(result["status"], "complete")
        jobs = [json.loads(x) for x in (output / "jobs.jsonl").read_text().splitlines()]
        self.assertNotIn("success", jobs[0])
        self.assertIn('"new_model_calls": 0', (output / "run.log").read_text())
        with self.assertRaises(FileExistsError):
            run(config, manifest, output, "caption", native_check=True)

    def test_official_reward_preprocessing_and_parser(self):
        code = '''
import json,sys
from batch_utils import core
c=core.Collage([sys.argv[1],sys.argv[1]])
indices=core.terminal_frame_indices(c.num_frames,c.fps)
frames=c.frames(indices)
assert len(frames)==20
assert core.parse_terminal("Score: 5")[0]==5
windows=core.dense_windows(c.num_frames,c.fps)
answers=[core.parse_dense("Worst: 3\\nQuality:\\n"+"3"*32+"\\nRationales:") for _ in windows]
assert all(a.valid for a in answers)
scores,_,_=core.stitch_dense(windows,answers,c.num_frames,4)
assert len(scores)==60 and set(scores)=={3}
assert not core.parse_dense("malformed").valid
print(json.dumps({"sampled_frames":len(frames),"windows":len(windows),"native_frames":len(scores)}))
'''
        result = subprocess.run([sys.executable, "-c", code, str(self.source)], cwd=VENDOR / "rewarder",
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_official_three_view_input_check(self):
        code = '''
import sys
from batch_utils import DataEnginePipeline
result=DataEnginePipeline().inspect_inputs(sys.argv[1],profile='dex',left_wrist=sys.argv[1],right_wrist=sys.argv[1])
assert result['new_model_calls']==0 and len(result['cameras'])==3
'''
        result = subprocess.run([sys.executable, "-c", code, str(self.source)], cwd=VENDOR / "data_engine",
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def group(self):
        import yaml

        path = self.root / "group.yaml"
        path.write_text(yaml.safe_dump({"robot_type": "single_arm", "dataset_root": "unused",
                                       "tasks": [{"name": "pick_bucket", "dataset": "pick_bucket",
                                                  "prompt": "Put food in the bucket, then lift it.",
                                                  "cameras": {"base": "front"}}]}))
        return str(path)

    def test_rollout_labels_and_no_video_failure(self):
        directory = self.root / "attempt/videos"
        directory.mkdir(parents=True)
        write_jsonl(directory / "episodes.jsonl", [{"episode_id": 0, "success": False}])
        config = {"task_group": self.group(), "tasks": ["pick_bucket"],
                  "source": {"kind": "rollout", "root": str(directory)}}
        with self.assertRaises(FileNotFoundError):
            rollout_rows(config)
        (directory / "episode_000_front.mp4").symlink_to(self.source)
        self.assertFalse(rollout_rows(config)[0]["success"])

    def test_lerobot_v2(self):
        dataset = self.root / "data/pick_bucket"
        (dataset / "meta").mkdir(parents=True)
        info = {"codebase_version": "v2.1", "fps": 30, "chunks_size": 1000,
                "features": {"observation.images.front": {"dtype": "video"}},
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
        write_json(dataset / "meta/info.json", info)
        write_jsonl(dataset / "meta/episodes.jsonl", [{"episode_index": 7, "length": 60}])
        path = dataset / "videos/chunk-000/observation.images.front/episode_000007.mp4"
        path.parent.mkdir(parents=True)
        path.symlink_to(self.source)
        config = {"task_group": self.group(), "source": {"kind": "lerobot", "root": str(dataset.parent)}}
        result = prepare(config, self.root / "prepared")
        row = read_manifest(result["manifest"])[0]
        self.assertEqual(row["episode_index"], 7)
        self.assertIsNone(row["success"])
        inspect_episode(row)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "requires pyarrow")
    def test_lerobot_v3_shared_video(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        dataset = self.root / "data/pick_bucket"
        (dataset / "meta/episodes/chunk-000").mkdir(parents=True)
        info = {"codebase_version": "v3.0", "fps": 30,
                "features": {"observation.images.front": {"dtype": "video"}},
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
        write_json(dataset / "meta/info.json", info)
        prefix = "videos/observation.images.front/"
        record = {"episode_index": 1, "length": 30, prefix+"chunk_index": 0, prefix+"file_index": 0,
                  prefix+"from_timestamp": 1.0, prefix+"to_timestamp": 2.0}
        pq.write_table(pa.Table.from_pylist([record]), dataset / "meta/episodes/chunk-000/file-000.parquet")
        path = dataset / "videos/observation.images.front/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True)
        path.symlink_to(self.source)
        config = {"task_group": self.group(), "source": {"kind": "lerobot", "root": str(dataset.parent)}}
        result = prepare(config, self.root / "prepared")
        row = read_manifest(result["manifest"])[0]
        inspect_episode(row)
        self.assertEqual(row["source"]["cameras"]["base"]["start_s"], 1.0)
        self.assertEqual(probe(row["cameras"]["base"])["frames"], 30)


if __name__ == "__main__":
    unittest.main()
