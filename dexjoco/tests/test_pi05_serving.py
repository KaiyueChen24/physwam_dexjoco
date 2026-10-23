"""CPU regression checks for cold-start serving and asynchronous evaluation failures."""
import asyncio
from collections import deque
import importlib.util
from pathlib import Path
from queue import Queue
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi_client import msgpack_numpy
import websockets.asyncio.server
import websockets.sync.client

ROOT = Path(__file__).resolve().parents[1]


def test_slow_inference_preserves_heartbeat_and_serializes_shared_policy():
    class Policy:
        active = 0
        max_active = 0
        def infer(self, obs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.3)  # Longer than the client's heartbeat deadline.
                return {"actions": np.ones((2, 22), dtype=np.float32)}
            finally:
                self.active -= 1

    policy = Policy()
    async def scenario():
        server = WebsocketPolicyServer(policy, metadata={"fixture": True})
        async with websockets.asyncio.server.serve(server._handler, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            def request():
                # A separate thread makes the client's heartbeat independent of the server loop.
                with websockets.sync.client.connect(
                    f"ws://127.0.0.1:{port}", ping_interval=0.02, ping_timeout=0.1
                ) as client:
                    assert msgpack_numpy.unpackb(client.recv())["fixture"]
                    client.send(msgpack_numpy.packb({"state": np.zeros(23)}))
                    response = msgpack_numpy.unpackb(client.recv(timeout=3))
                    assert response["actions"].shape == (2, 22)
            await asyncio.gather(asyncio.to_thread(request), asyncio.to_thread(request))
        assert policy.max_active == 1
    asyncio.run(scenario())


@pytest.fixture
def evaluator(monkeypatch):
    # The wait/queue behavior doesn't require MuJoCo, rendering, or a GPU.
    env = ModuleType("serving_fixture.dexjoco_openpi_env")
    env.DexJoCoOpenPIEnv = object
    monkeypatch.setitem(sys.modules, env.__name__, env)
    spec = importlib.util.spec_from_file_location(
        "serving_fixture.eval", ROOT / "dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_cold_start_wait_preserves_timestamp_zero_actions(evaluator):
    queue, buffer = Queue(), deque()
    worker = SimpleNamespace(is_alive=lambda: True)
    actions = np.zeros((30, 22), dtype=np.float32)
    producer = threading.Thread(target=lambda: (
        time.sleep(0.03), queue.put(evaluator.ActionChunk(actions, timestamp=0))
    ))
    producer.start()
    try:
        evaluator._wait_for_initial_actions(queue, buffer, worker, False, timeout=1)
    finally:
        producer.join()
    assert [action.timestamp for action in buffer] == list(range(30))
    np.testing.assert_array_equal(buffer[0].action, actions[0])


def test_dead_worker_fails_initial_wait_and_episode_drain(evaluator):
    worker = SimpleNamespace(is_alive=lambda: False, exitcode=1)
    event = threading.Event()
    event.set()
    with pytest.raises(RuntimeError, match="Inference worker exited.*exitcode=1"):
        evaluator._wait_for_initial_actions(Queue(), deque(), worker, False)
    with pytest.raises(RuntimeError, match="Inference worker exited"):
        evaluator._wait_for_inference(event, worker)


def test_unresponsive_worker_has_bounded_wait(evaluator):
    worker = SimpleNamespace(is_alive=lambda: True)
    with pytest.raises(TimeoutError, match="first policy action"):
        evaluator._wait_for_initial_actions(Queue(), deque(), worker, False, timeout=0)
    event = threading.Event()
    event.set()
    with pytest.raises(TimeoutError, match="pending inference"):
        evaluator._wait_for_inference(event, worker, timeout=0)


_PARENT_RENDERER_SENTINEL = None


def _probe_spawn_worker(obs_queue, action_queue, stop_event, port, inferencing_event, seed, host, metrics):
    # The parent changes this global after imports, like a live native context.
    # A fork child inherits it; a spawn child imports a fresh module instead.
    for episode in range(2):
        obs = obs_queue.get(timeout=10)
        inferencing_event.clear()
        action_queue.put((obs["episode"], obs["image"], _PARENT_RENDERER_SENTINEL is None))
    stop_event.wait(10)


def test_inference_worker_isolates_parent_resources_and_keeps_ipc_across_episodes(evaluator, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_PARENT_RENDERER_SENTINEL", object())
    monkeypatch.setattr(evaluator, "inference_process", _probe_spawn_worker)
    observations, actions, stop, pending, worker = evaluator._create_inference_worker(8000, 0, "127.0.0.1")
    try:
        worker.start()
        for episode in range(2):
            image = np.full((224, 224, 3), episode, dtype=np.uint8)
            pending.set()
            observations.put({"episode": episode, "image": image})
            received_episode, received_image, isolated = actions.get(timeout=10)
            assert received_episode == episode
            np.testing.assert_array_equal(received_image, image)
            assert isolated, "worker inherited parent renderer state via fork"
            assert not pending.is_set(), "spawn worker event must be shared with parent"
        stop.set()
        worker.join(timeout=10)
        assert worker.exitcode == 0
    finally:
        stop.set()
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2)
        for queue in (observations, actions):
            queue.cancel_join_thread()
            queue.close()


@pytest.mark.parametrize("dual_arm,dimension", [(False, 22), (True, 44)])
@pytest.mark.parametrize("old_timestamp", [0, 999])
def test_late_previous_episode_chunk_is_discarded_before_timestamp_check(
    evaluator, dual_arm, dimension, old_timestamp
):
    queue, buffer = Queue(), deque()
    old = np.full((30, dimension), 50.0, dtype=np.float32)
    current = np.zeros((30, dimension), dtype=np.float32)
    queue.put(evaluator.ActionChunk(old, old_timestamp, episode_id=16))
    queue.put(evaluator.ActionChunk(current, 0, episode_id=17))
    evaluator.receive_actions(queue, buffer, 0, dual_arm, episode_id=17)
    assert [action.timestamp for action in buffer] == list(range(30))
    assert all(action.episode_id == 17 for action in buffer)
    np.testing.assert_array_equal(np.stack([action.action for action in buffer]), current)


def test_initial_wait_tolerates_delayed_old_response_after_pending_is_clear(evaluator):
    queue, buffer = Queue(), deque()
    worker = SimpleNamespace(is_alive=lambda: True)
    event = threading.Event()  # Old request is complete, but its feeder is delayed.
    actions = np.zeros((30, 22), dtype=np.float32)

    def deliver():
        time.sleep(0.03)
        queue.put(evaluator.ActionChunk(actions + 50, 999, episode_id=16))
        queue.put(evaluator.ActionChunk(actions, 0, episode_id=17))

    producer = threading.Thread(target=deliver)
    producer.start()
    try:
        evaluator._wait_for_inference(event, worker, timeout=1)
        evaluator._wait_for_initial_actions(queue, buffer, worker, False, timeout=1, episode_id=17)
    finally:
        producer.join()
    np.testing.assert_array_equal(np.stack([action.action for action in buffer]), actions)


def test_current_episode_future_timestamp_still_rejected(evaluator):
    queue = Queue()
    queue.put(evaluator.ActionChunk(np.zeros((30, 22)), 1, episode_id=17))
    with pytest.raises(AssertionError):
        evaluator.receive_actions(queue, deque(), 0, False, episode_id=17)


def test_worker_echoes_episode_id_to_actions_and_metrics(evaluator, monkeypatch, tmp_path):
    observations, actions = Queue(), Queue()
    stop, pending = threading.Event(), threading.Event()
    pending.set()
    observations.put(evaluator.Observation({"state": np.zeros(23)}, 4, episode_id=17))

    class Client:
        def infer(self, observation):
            stop.set()
            return {"actions": np.zeros((30, 22)), "server_timing": {"infer_ms": 1.0}}

    monkeypatch.setattr(evaluator.websocket_client_policy, "WebsocketClientPolicy", lambda **kwargs: Client())
    monkeypatch.setattr(evaluator.signal, "signal", lambda *args: None)
    metrics = tmp_path / "latency.jsonl"
    evaluator.inference_process(observations, actions, stop, 8000, pending, 0, "127.0.0.1", metrics)
    chunk = actions.get_nowait()
    assert (chunk.episode_id, chunk.timestamp) == (17, 4)
    assert not pending.is_set()
    import json
    assert json.loads(metrics.read_text())["episode_id"] == 17
