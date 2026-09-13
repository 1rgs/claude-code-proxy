"""Regression tests: the proxy must drop its upstream request when its client goes away.

Run: uv run python -m unittest test_client_disconnect -v

A fake OpenAI-compatible upstream runs in this process; the proxy runs as a
real uvicorn subprocess pointed at it. No API keys or network access needed.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM_WAIT = 30  # the first request through litellm can take a while to go out
CLOSE_WAIT = 5
MODEL = "claude-3-5-sonnet-20241022"  # routed to BIG_MODEL


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def completion_chunk(content, finish_reason):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4.1",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
    }


class FakeUpstream:
    """OpenAI-compatible endpoint whose mode is "hang", "json" or "sse".

    "hang" never answers, like llama-server reading a long prompt, and
    records whether the proxy hangs up on it."""

    def __init__(self):
        self.port = free_port()
        self.mode = "hang"
        self.generation = 0
        self.received = threading.Event()
        self.client_closed = threading.Event()
        self._hanging = []
        self.loop = asyncio.new_event_loop()
        ready = threading.Event()
        threading.Thread(target=self._serve, args=(ready,), daemon=True).start()
        ready.wait()

    def reset(self, mode):
        asyncio.run_coroutine_threadsafe(self._reset(mode), self.loop).result()

    def stop(self):
        self.reset("hang")
        self.loop.call_soon_threadsafe(self.loop.stop)

    async def _reset(self, mode):
        # Bump the generation first so connections closed here can't count
        # as a hang-up in the next test.
        self.generation += 1
        for writer in self._hanging:
            writer.close()
        self._hanging.clear()
        self.mode = mode
        self.received.clear()
        self.client_closed.clear()

    def _serve(self, ready):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(asyncio.start_server(self._handle, "127.0.0.1", self.port))
        ready.set()
        self.loop.run_forever()

    async def _handle(self, reader, writer):
        generation = self.generation
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            length = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":", 1)[1])
            await reader.readexactly(length)
        except (asyncio.IncompleteReadError, ConnectionError):
            return
        self.received.set()

        if self.mode == "hang":
            self._hanging.append(writer)
            try:
                closed = await reader.read(1) == b""
            except ConnectionError:
                closed = True
            if closed and generation == self.generation:
                self.client_closed.set()
            return

        if self.mode == "json":
            body = json.dumps({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4.1",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            head = b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: %d\r\nconnection: close\r\n\r\n" % len(body)
        else:
            events = [completion_chunk("pong", None), completion_chunk("", "stop")]
            body = ("".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n").encode()
            head = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n"
        writer.write(head + body)
        await writer.drain()
        writer.close()


class ClientDisconnectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = FakeUpstream()
        cls.port = free_port()
        cls.log_path = tempfile.mkstemp(prefix="proxy-test-", suffix=".log")[1]
        env = {
            **os.environ,
            "PREFERRED_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": f"http://127.0.0.1:{cls.upstream.port}/v1",
            "BIG_MODEL": "gpt-4.1",
            "SMALL_MODEL": "gpt-4.1",
        }
        with open(cls.log_path, "w") as log:
            cls.proxy = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(cls.port)],
                cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{cls.port}/", timeout=1).close()
                return
            except OSError:
                time.sleep(0.2)
        cls.tearDownClass()
        raise RuntimeError(f"proxy did not start; see {cls.log_path}")

    @classmethod
    def tearDownClass(cls):
        cls.proxy.kill()
        cls.proxy.wait()
        cls.upstream.stop()

    def request_body(self, stream):
        return json.dumps({
            "model": MODEL,
            "max_tokens": 16,
            "stream": stream,
            "messages": [{"role": "user", "content": "ping"}],
        }).encode()

    def post(self, stream):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/messages",
            data=self.request_body(stream),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode()

    def post_then_hang_up(self, stream):
        body = self.request_body(stream)
        with socket.create_connection(("127.0.0.1", self.port)) as sock:
            sock.sendall(
                b"POST /v1/messages HTTP/1.1\r\nhost: localhost\r\ncontent-type: application/json\r\n"
                b"content-length: %d\r\n\r\n" % len(body) + body
            )
            self.assertTrue(self.upstream.received.wait(UPSTREAM_WAIT), "request never reached upstream")
        # Leaving the with-block closes the socket: the client is gone.

    def assert_upstream_hung_up(self):
        self.assertTrue(
            self.upstream.client_closed.wait(CLOSE_WAIT),
            f"proxy kept its upstream request open {CLOSE_WAIT}s after the client left; see {self.log_path}",
        )

    def test_streaming_request_is_cancelled_upstream_when_client_disconnects(self):
        self.upstream.reset("hang")
        self.post_then_hang_up(stream=True)
        self.assert_upstream_hung_up()

    def test_non_streaming_request_is_cancelled_upstream_when_client_disconnects(self):
        self.upstream.reset("hang")
        self.post_then_hang_up(stream=False)
        self.assert_upstream_hung_up()

    def test_streaming_request_still_streams_the_upstream_answer(self):
        self.upstream.reset("sse")
        self.assertIn("pong", self.post(stream=True))

    def test_non_streaming_request_still_returns_the_upstream_answer(self):
        self.upstream.reset("json")
        reply = json.loads(self.post(stream=False))
        self.assertEqual(reply["content"][0]["text"], "pong")


if __name__ == "__main__":
    unittest.main()
