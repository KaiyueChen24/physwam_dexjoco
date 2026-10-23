"""Launch the two upstream packages in separate subprocesses/environments."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

from .common import VENDOR, WORKSPACE, read_jsonl, read_manifest, resolve_path, sha256, verify_vendor, write_json, write_jsonl
from .jobs import build_jobs
from .media import inspect_episode

STAGES = ("terminal", "dense", "rubric", "caption")


def verify_results(output, stage, ids):
    if stage == "caption":
        for name in ids:
            result = json.loads((output / "annotations" / f"{name}.json").read_text())
            if not isinstance(result.get("segments"), list):
                raise ValueError(f"{name}: official caption output has no segments list")
    else:
        records = read_jsonl(output / f"{stage}.jsonl")
        actual = [row["id"] for row in records]
        if len(actual) != len(set(actual)) or set(actual) != set(ids):
            raise ValueError("Backend results do not cover the exact input episode ids")


def make_plan(config, manifest, output, stage, gpus=None, native_check=False):
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage}")
    rows = read_manifest(manifest)
    jobs, labels = build_jobs(rows, config, stage)
    output = Path(output).resolve()
    runtime = config.get("runtime", {})
    gpu_list = str(gpus if gpus is not None else runtime.get("gpus", "0"))
    devices = [x.strip() for x in gpu_list.split(",")]
    if not all(x.isdigit() for x in devices) or len(devices) != len(set(devices)):
        raise ValueError("gpus must be distinct comma-separated physical GPU indices")
    family = "captioner" if stage == "caption" else "rewarder"
    python = runtime.get(f"{family}_python")
    # Preserve the venv's symlink path: resolving it to the system binary loses
    # pyvenv.cfg discovery and silently selects the wrong dependency environment.
    python = os.path.abspath(WORKSPACE / Path(python).expanduser()) if python else sys.executable
    cwd = VENDOR / ("data_engine" if stage == "caption" else "rewarder")
    command = [python, "-m"]
    if stage == "caption":
        command += ["batch_utils", "--input-jsonl", str(output / "jobs.jsonl"),
                    "--output-dir", str(output / "annotations")]
        if config.get("caption", {}).get("include_audit", True):
            command += ["--include-audit"]
        if native_check:
            command += ["--validate-only"]
    else:
        settings = config.get("rewarder", {})
        tp = settings.get("rubric_tensor_parallel_size", 2) if stage == "rubric" else settings.get("tensor_parallel_size", 1)
        if not isinstance(tp, int) or tp < 1 or tp > len(devices):
            raise ValueError(f"tensor_parallel_size={tp} needs that many visible GPUs; got {gpu_list}")
        source = json.loads((VENDOR / "sources.lock.json").read_text())["sources"]["rewarder"]
        model = settings.get("model_path")
        command += [f"batch_utils.{stage}_reward", "--jobs", str(output / "jobs.jsonl"),
                    "--output", str(output / f"{stage}.jsonl"),
                    "--model", str(resolve_path(model)) if model else source["repo_id"]]
        if not model:
            command += ["--revision", source["revision"]]
        sequences = settings.get("max_num_seqs", 1)
        utilization = settings.get("gpu_memory_utilization", 0.85)
        if not isinstance(sequences, int) or sequences < 1 or not 0 < utilization < 1:
            raise ValueError("Invalid max_num_seqs/gpu_memory_utilization")
        command += ["--tensor-parallel-size", str(tp), "--max-num-seqs", str(sequences),
                    "--gpu-memory-utilization", str(utilization), "--seed", str(settings.get("seed", 0))]
        if stage == "terminal" and settings.get("with_rationale", True):
            command += ["--with-rationale"]
        if stage == "dense" and settings.get("frame_scores", False):
            command += ["--frame-scores"]
        if stage == "rubric":
            samples = config.get("rubric", {}).get("num_samples", 9)
            if not isinstance(samples, int) or samples < 1:
                raise ValueError("rubric.num_samples must be positive")
            command += ["--num-samples", str(samples)]
            if config.get("rubric", {}).get("keep_thinking", False):
                command += ["--keep-thinking"]
        if native_check:
            command += ["--dry-run"]
    env = {"CUDA_VISIBLE_DEVICES": "" if native_check else ",".join(devices),
           "PYTHONPATH": str(cwd), "PYTHONUNBUFFERED": "1",
           "FLORA_MODEL_CACHE": str(resolve_path(runtime.get("model_cache", "flora/models/data_engine"))),
           "FLORA_TEMPSEG_REPO": "nvidia/Flora-TempSeg", "FLORA_CAPTIONER_REPO": "nvidia/Flora-Captioner"}
    if runtime.get("hf_home"):
        env["HF_HOME"] = str(resolve_path(runtime["hf_home"]))
    return {"stage": stage, "mode": "native_check" if native_check else "inference", "cwd": str(cwd),
            "argv": command, "shell_preview": shlex.join(command), "env": env,
            "output": str(output), "episodes": len(rows)}, jobs, labels, rows


def run(config, manifest, output, stage, *, gpus=None, dry_run=False, native_check=False):
    plan, jobs, labels, rows = make_plan(config, manifest, output, stage, gpus, native_check)
    if dry_run:
        return plan
    output = Path(plan["output"])
    if output.exists():
        raise FileExistsError(f"Use a new run directory; partial runs are preserved: {output}")
    if not Path(plan["argv"][0]).is_file():
        raise FileNotFoundError(f"Install the {stage} runtime first: {plan['argv'][0]}")
    sources = verify_vendor()
    # Check all episodes before a backend can download weights or allocate CUDA.
    inspections = [inspect_episode(row, stage, config.get("caption", {}).get("views", "auto")) for row in rows]
    output.mkdir(parents=True)
    write_jsonl(output / "jobs.jsonl", jobs)
    write_jsonl(output / "labels.jsonl", labels)
    write_json(output / "inputs.json", inspections)
    write_json(output / "run.json", {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                                    "plan": plan, "config": config, "sources": sources,
                                    "manifest": str(Path(manifest).resolve()), "manifest_sha256": sha256(manifest),
                                    "episodes": rows})
    env = {**os.environ, **plan["env"]}
    # Popen, not shell=True: instructions and paths never become shell code.
    with (output / "run.log").open("x") as log:
        process = subprocess.Popen(plan["argv"], cwd=plan["cwd"], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   start_new_session=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            write_json(output / "status.json", {"status": "interrupted"})
            raise
        finally:
            process.stdout.close()
    if code:
        write_json(output / "status.json", {"status": "failed", "returncode": code})
        raise RuntimeError(f"Official {stage} backend exited with {code}; see {output / 'run.log'}")
    try:
        if not native_check:
            verify_results(output, stage, [r["id"] for r in rows])
    except (OSError, ValueError, KeyError, TypeError) as error:
        write_json(output / "status.json", {"status": "failed", "returncode": code, "error": str(error)})
        raise
    write_json(output / "status.json", {"status": "complete", "returncode": code})
    return {"output": str(output), "stage": stage, "mode": plan["mode"], "status": "complete"}
