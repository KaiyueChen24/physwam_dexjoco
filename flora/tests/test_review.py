"""Exercise the exported report against actual media and hostile caption text."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from physwam_flora.common import sha256, write_json, write_jsonl
from physwam_flora.media import probe
from physwam_flora.review import build_review

HAS_MEDIA = all(importlib.util.find_spec(name) for name in ("av", "numpy"))


def fixture(root):
    from test_pipeline import video

    source = root / "source.mp4"
    video(source, frames=30)
    run = root / "synthetic_caption"
    run.mkdir()
    row = {"id": "sample", "instruction": "Synthetic fixture, not model predictions."}
    write_json(run / "run.json", {"plan": {"stage": "caption", "mode": "inference"}, "episodes": [row]})
    write_json(run / "status.json", {"status": "complete"})
    write_jsonl(run / "jobs.jsonl", [{"id": "sample", "video": str(source),
                                      "left_wrist": str(source), "right_wrist": str(source)}])
    annotation = {"duration_s": 1.0, "input_sha256": {role: sha256(source) for role in ("main", "wrist_left", "wrist_right")},
                  "segments": [{"start_s": 0, "end_s": .7, "caption": "</script><script>window.injected=true</script>", "status": "complete"},
                               {"start_s": .3, "end_s": 1, "caption": "", "status": "unsupported_target_frames"}]}
    (run / "annotations").mkdir()
    write_json(run / "annotations/sample.json", annotation)
    return run, annotation


@unittest.skipUnless(HAS_MEDIA, "PyAV and numpy required")
class Review(unittest.TestCase):
    def test_portable_report_preserves_frames_and_escapes_captions(self):
        import av

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, annotation = fixture(root)
            output = root / "review"
            result = build_review(run, output)
            self.assertEqual(result["episodes"], 1)
            report = json.loads((output / "report.json").read_text())
            episode = report["episodes"][0]
            # Preserve overlapping intervals and failed/empty captions for review.
            self.assertEqual(len(episode["segments"]), 2)
            self.assertEqual(episode["segments"][1]["status"], "unsupported_target_frames")
            for relative in episode["videos"].values():
                self.assertFalse(Path(relative).is_absolute())
                self.assertEqual(probe(output / relative)["frames"], 30)
                with av.open(str(output / relative)) as container:
                    stream = container.streams.video[0]
                    self.assertEqual(stream.codec_context.profile, "High")
                    self.assertEqual(stream.codec_context.pix_fmt, "yuv420p")
            page = (output / "index.html").read_text()
            self.assertNotIn(annotation["segments"][0]["caption"], page)
            self.assertIn("\\u003c/script>", page)
            with self.assertRaises(FileExistsError):
                build_review(run, output)

    def test_invalid_intervals_and_changed_inputs_leave_no_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, annotation = fixture(root)
            path = run / "annotations/sample.json"
            output = root / "review"
            annotation["segments"][0]["end_s"] = 2
            path.write_text(json.dumps(annotation))
            with self.assertRaisesRegex(ValueError, "interval"):
                build_review(run, output)
            self.assertFalse(output.exists())
            annotation["segments"][0]["end_s"] = .7
            annotation["input_sha256"]["main"] = "wrong"
            path.write_text(json.dumps(annotation))
            with self.assertRaisesRegex(ValueError, "source video changed"):
                build_review(run, output)
            self.assertFalse(output.exists())
