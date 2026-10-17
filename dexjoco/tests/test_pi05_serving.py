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
