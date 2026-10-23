"""CPU-only input checks and exact episode extraction from LeRobot v3 files."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path


def probe(path):
    import av

    times = []
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"{path}: expected exactly one video stream")
        stream = container.streams.video[0]
        rate = stream.average_rate
        if not rate or rate <= 0:
            raise ValueError(f"{path}: missing video frame rate")
        size = (stream.width, stream.height)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"{path}: missing presentation timestamp")
            times.append(Fraction(frame.pts) * stream.time_base)
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"{path}: need at least two frames with increasing timestamps")
    times = [t - times[0] for t in times]
    if any(abs(t - Fraction(i, rate)) > Fraction(1, 1_000_000) for i, t in enumerate(times)):
        raise ValueError(f"{path}: expected constant-rate video; variable-rate reward inputs are unsupported")
    return {"fps": float(rate), "frames": len(times), "width": size[0], "height": size[1],
            "duration_s": float(times[-1] + 1 / rate), "size_bytes": Path(path).stat().st_size}


def inspect_episode(row, stage="terminal", caption_views="auto"):
    cameras = row["cameras"]
    if stage == "caption":
        roles = ["base"]
        if caption_views == "three" or (caption_views == "auto" and "wrist_left" in cameras):
            roles += ["wrist_left", "wrist_right"]
    else:
        roles = list(cameras)
    info = {role: probe(cameras[role]) for role in roles}
    base = info["base"]
    for role, value in info.items():
        if value["frames"] != base["frames"] or abs(value["fps"] - base["fps"]) > 1e-6:
            raise ValueError(f"{row['id']}: unsynchronized {role} video")
        if stage == "caption" and (value["duration_s"] > 120.000001 or value["frames"] > 7200
                                   or value["fps"] < 4 or max(value["width"], value["height"]) > 4096
                                   or value["size_bytes"] > 256 * 1024 ** 2):
            raise ValueError(f"{row['id']}/{role}: exceeds official caption input limits; no automatic truncation")
    if row.get("fps") is not None and abs(row["fps"] - base["fps"]) > 1e-6:
        raise ValueError(f"{row['id']}: manifest fps disagrees with video")
    if row.get("frames") is not None and row["frames"] != base["frames"]:
        raise ValueError(f"{row['id']}: episode length disagrees with video")
    return {"id": row["id"], "cameras": info}


def extract_episode(source, target, start_s, end_s, fps, expected_frames):
    """Re-encode only the v3 episode, without dropping/resampling its frames.

    H.264 CRF 0 avoids introducing lossy compression; start/end are absolute file
    timestamps. Export clock is episode-relative and source offsets stay in the manifest.
    """
    import av

    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    if not 0 <= start_s < end_s or expected_frames < 2:
        raise ValueError("Invalid episode interval or frame count")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial.mp4")
    rate = Fraction(str(fps)).limit_denominator(100000)
    count = 0
    first_time = None
    try:
        with av.open(str(source)) as incoming, av.open(str(temporary), "w") as outgoing:
            stream = incoming.streams.video[0]
            if abs(float(stream.average_rate) - fps) > 1e-6:
                raise ValueError(f"{source}: source rate differs from LeRobot metadata")
            encoder = outgoing.add_stream("libx264", rate=rate)
            encoder.width, encoder.height = stream.width, stream.height
            if encoder.width % 2 or encoder.height % 2:
                raise ValueError("v3 export requires even dimensions for yuv420p")
            encoder.pix_fmt = "yuv420p"
            encoder.options = {"crf": "0", "preset": "fast"}
            # Seek to the keyframe preceding the episode, then select by actual PTS.
            incoming.seek(max(0, int(start_s / stream.time_base)), stream=stream, backward=True)
            for frame in incoming.decode(stream):
                if frame.pts is None:
                    raise ValueError("v3 input frame is missing its timestamp")
                time = float(frame.pts * stream.time_base)
                if time < start_s - 1e-6:
                    continue
                if time >= end_s - 1e-6:
                    break
                first_time = time if first_time is None else first_time
                if abs(time - (first_time + count / fps)) > 1e-5:
                    raise ValueError("v3 episode is not constant-rate; refusing temporal resampling")
                frame.pts, frame.time_base = count, 1 / rate
                for packet in encoder.encode(frame):
                    outgoing.mux(packet)
                count += 1
            for packet in encoder.encode():
                outgoing.mux(packet)
        if count != expected_frames or first_time is None or abs(first_time - start_s) > 1e-5:
            raise ValueError(f"{source}: v3 interval yielded {count} frames, expected {expected_frames}, first={first_time}")
        temporary.rename(target)
    finally:
        temporary.unlink(missing_ok=True)
