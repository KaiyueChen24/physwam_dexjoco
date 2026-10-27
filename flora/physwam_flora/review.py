"""Build a portable caption review page with browser-compatible video copies."""
from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
import re
import tempfile

from .common import read_jsonl, sha256, write_json
from .media import probe


def preview_video(source, target):
    """Keep the native frame clock; the lossy H.264 copy is for viewing only."""
    import av

    original = probe(source)
    rate = Fraction(str(original["fps"])).limit_denominator(100000)
    target.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(source)) as incoming, av.open(str(target), "w", options={"movflags": "+faststart"}) as outgoing:
        stream = incoming.streams.video[0]
        if stream.width % 2 or stream.height % 2:
            raise ValueError("Review previews require even video dimensions")
        encoder = outgoing.add_stream("libx264", rate=rate)
        encoder.width, encoder.height, encoder.pix_fmt = stream.width, stream.height, "yuv420p"
        encoder.options = {"crf": "18", "preset": "fast", "profile": "high"}
        for index, frame in enumerate(incoming.decode(stream)):
            frame.pts, frame.time_base = index, 1 / rate
            for packet in encoder.encode(frame):
                outgoing.mux(packet)
        for packet in encoder.encode():
            outgoing.mux(packet)
    exported = probe(target)
    if any(original[key] != exported[key] for key in ("frames", "width", "height")) or abs(original["fps"] - exported["fps"]) > 1e-6:
        raise ValueError("Preview export changed the frame clock or dimensions")
    return exported


def review_segments(annotation):
    duration = annotation.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Caption annotation needs a positive finite duration_s")
    segments = annotation.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Caption annotation needs a segments list")
    result = []
    for index, segment in enumerate(segments):
        start, end = segment.get("start_s"), segment.get("end_s")
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in (start, end)):
            raise ValueError(f"Segment {index}: timestamps must be finite numbers")
        if not 0 <= start < end <= duration + 0.001:
            raise ValueError(f"Segment {index}: interval outside episode duration")
        caption, status = segment.get("caption", ""), segment.get("status", "unknown")
        if not isinstance(caption, str) or not isinstance(status, str):
            raise ValueError(f"Segment {index}: caption/status must be strings")
        result.append({"index": index, "start_s": start, "end_s": end, "caption": caption,
                       "status": status, "segmentation_provenance": segment.get("segmentation_provenance")})
    return result


def build_review(run_dir, output, episode_id=None):
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new review directory: {output}")
    record = json.loads((run_dir / "run.json").read_text())
    status = json.loads((run_dir / "status.json").read_text())
    if record["plan"]["stage"] != "caption" or record["plan"]["mode"] != "inference" or status.get("status") != "complete":
        raise ValueError("Review requires a completed caption inference run")
    rows = {row["id"]: row for row in record["episodes"]}
    jobs = read_jsonl(run_dir / "jobs.jsonl")
    if episode_id is not None:
        jobs = [job for job in jobs if job["id"] == episode_id]
    if not jobs:
        raise ValueError("No matching caption episodes")
    report = {"schema_version": 1, "run_name": run_dir.name, "episodes": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Failed exports never leave an apparently complete review directory.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for job in jobs:
            name = job["id"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name):
                raise ValueError("Invalid episode id")
            annotation_path = run_dir / "annotations" / f"{name}.json"
            annotation = json.loads(annotation_path.read_text())
            segments = review_segments(annotation)
            cameras = {"base": job["video"]}
            cameras.update({key: job[field] for key, field in (("wrist_left", "left_wrist"), ("wrist_right", "right_wrist")) if job.get(field)})
            previews, media = {}, {}
            for role, value in cameras.items():
                source = Path(value)
                if not source.is_absolute():
                    source = run_dir / source
                expected = annotation.get("input_sha256", {}).get("main" if role == "base" else role)
                if expected and sha256(source) != expected:
                    raise ValueError(f"{name}/{role}: source video changed since inference")
                relative = f"videos/{name}/{role}.mp4"
                print(f"Exporting review preview: {name}/{role}", flush=True)
                media[role] = preview_video(source, staging / relative)
                previews[role] = relative
            first = media["base"]
            for role, info in media.items():
                if info["frames"] != first["frames"] or abs(info["fps"] - first["fps"]) > 1e-6:
                    raise ValueError(f"{name}/{role}: unsynchronized review videos")
            if abs(first["duration_s"] - annotation["duration_s"]) > max(0.001, 1 / first["fps"]):
                raise ValueError(f"{name}: annotation duration differs from video")
            report["episodes"].append({"id": name, "instruction": rows[name]["instruction"],
                                       "annotation_sha256": sha256(annotation_path),
                                       "duration_s": annotation["duration_s"], "segments": segments,
                                       "videos": previews, "media": media})
        write_json(staging / "report.json", report)
        payload = json.dumps(report, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
        template = Path(__file__).with_name("review.html").read_text()
        (staging / "index.html").write_text(template.replace("__FLORA_REPORT_JSON__", payload))
        staging.rename(output)
    return {"output": str(output), "page": str(output / "index.html"), "episodes": len(report["episodes"]),
            "note": "Viewing copies only; predictions unchanged. Human feedback is not benchmark accuracy."}
