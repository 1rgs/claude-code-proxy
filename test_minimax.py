"""Offline tests through FastAPI, real LiteLLM, and mocked HTTP wire transport."""
import importlib
import json
import os
import unittest
from unittest.mock import patch

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["PREFERRED_PROVIDER"] = "minimax"
os.environ["MINIMAX_API_KEY"] = "offline-test-key"
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import httpx
from fastapi.testclient import TestClient
from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
import server


class MiniMaxTransportTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "PREFERRED_PROVIDER": "minimax",
            "MINIMAX_API_KEY": "offline-test-key",
            "MINIMAX_API_STYLE": "anthropic",
            "MINIMAX_BASE_URL": "https://api.minimax.io/v1",
            "MINIMAX_ANTHROPIC_BASE_URL": "https://api.minimax.io/anthropic",
        })
        self.env.start()
        for name in ("BIG_MODEL", "SMALL_MODEL"):
            os.environ.pop(name, None)
        importlib.reload(server)
        self.client = TestClient(server.app)
        self.calls = []
        self.sync_patch = patch.object(httpx.HTTPTransport, "handle_request", self.transport)
        self.async_patch = patch.object(httpx.AsyncHTTPTransport, "handle_async_request", self.async_transport)
        self.sync_patch.start()
        self.async_patch.start()
        self.aiohttp_patch = patch.object(LiteLLMAiohttpTransport, "handle_async_request", self.async_transport)
        self.aiohttp_patch.start()
        self.network_patch = patch("socket.socket.connect", side_effect=AssertionError("Unexpected network access"))
        self.network_patch.start()

    def tearDown(self):
        self.network_patch.stop()
        self.aiohttp_patch.stop()
        self.client.close()
        self.async_patch.stop()
        self.sync_patch.stop()
        self.env.stop()

    async def async_transport(self, request):
        return self.transport(request)

    def transport(self, request):
        body = json.loads(request.content)
        self.calls.append((request, body))
        model = body["model"]
        if getattr(self, "tool_fixture", False):
            return self.tool_response(request, body)
        if request.url.path.endswith("/chat/completions"):
            response = {
                "id": "chatcmpl-offline", "object": "chat.completion",
                "created": 1, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
            if body.get("stream"):
                chunks = [
                    {"id": "chatcmpl-offline", "object": "chat.completion.chunk", "created": 1, "model": model,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}]},
                    {"id": "chatcmpl-offline", "object": "chat.completion.chunk", "created": 1, "model": model,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
                data = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
                return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"}, request=request)
        else:
            self.assertEqual(request.url.path, "/anthropic/v1/messages")
            response = {
                "id": "msg_offline", "type": "message", "role": "assistant", "model": model,
                "content": [{"type": "text", "text": "Hello"}], "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
            if body.get("stream"):
                events = [
                    {"type": "message_start", "message": {**response, "content": [], "stop_reason": None}},
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
                    {"type": "content_block_stop", "index": 0},
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 2}},
                    {"type": "message_stop"},
                ]
                data = "".join("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n" for event in events)
                return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"}, request=request)
        return httpx.Response(200, json=response, request=request)

    def send(self, model="claude-sonnet", **options):
        before = len(self.calls)
        response = self.client.post("/v1/messages", json={
            "model": model, "max_tokens": 524288,
            "messages": [{"role": "user", "content": "Hello"}], **options,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.calls), before + 1)
        if options.get("stream"):
            self.assertIn("event: message_stop", response.text)
            self.assertIn("Hello", response.text)
        else:
            self.assertEqual(response.json()["content"][0]["text"], "Hello")
        return self.calls[-1]

    def tool_response(self, request, body):
        openai = request.url.path.endswith("/chat/completions")
        has_results = any(message["role"] == "tool" or any(
            block.get("type") == "tool_result" for block in (message.get("content") or []) if isinstance(block, dict)
        ) for message in body["messages"])
        thinking = {"type": "thinking", "thinking": "I should use both tools.", "signature": "opaque-signature-fixture"}
        tools = [{"type": "tool_use", "id": f"call_fixture_{i}", "name": "echo",
                  "input": {"text": f"value {i}", "nested": {"n": i}}} for i in range(2)]
        blocks = [{"type": "text", "text": "Both done."}] if has_results else [thinking, *tools]
        if openai:
            calls = [{"id": block["id"], "type": "function", "function": {
                "name": block["name"], "arguments": json.dumps(block["input"])
            }} for block in tools]
            message = {"role": "assistant", "content": "Both done."} if has_results else {
                "role": "assistant", "content": None, "reasoning_content": thinking["thinking"], "tool_calls": calls,
            }
            reason = "stop" if has_results else "tool_calls"
            response = {"id": "chatcmpl-fixture", "object": "chat.completion", "created": 1,
                        "model": body["model"], "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
            if body.get("stream"):
                deltas = [{"content": "Both done."}] if has_results else [
                    {"reasoning_content": thinking["thinking"]},
                    {"tool_calls": [{**call, "index": i, "function": {**call["function"], "arguments": ""}}
                                    for i, call in enumerate(calls)]},
                    *[{"tool_calls": [{"index": i, "function": {"arguments": call["function"]["arguments"][:12]}}
                                      for i, call in enumerate(calls)]},
                      {"tool_calls": [{"index": i, "function": {"arguments": call["function"]["arguments"][12:]}}
                                      for i, call in enumerate(calls)]}],
                ]
                chunks = [{"id": response["id"], "object": "chat.completion.chunk", "created": 1, "model": body["model"],
                           "choices": [{"index": 0, "delta": delta, "finish_reason": None}]} for delta in deltas]
                chunks.append({"id": response["id"], "object": "chat.completion.chunk", "created": 1, "model": body["model"],
                               "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
                data = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
                return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"}, request=request)
        else:
            response = {"id": "msg_fixture", "type": "message", "role": "assistant", "model": body["model"],
                        "content": blocks, "stop_reason": "end_turn" if has_results else "tool_use", "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 20}}
            if body.get("stream"):
                events = [{"type": "message_start", "message": {**response, "content": [], "stop_reason": None}}]
                for i, block in enumerate(blocks):
                    if block["type"] == "thinking":
                        start = {"type": "thinking", "thinking": "", "signature": ""}
                        deltas = [{"type": "thinking_delta", "thinking": block["thinking"]},
                                  {"type": "signature_delta", "signature": block["signature"]}]
                    elif block["type"] == "tool_use":
                        start = {**block, "input": {}}
                        arguments = json.dumps(block["input"])
                        deltas = [{"type": "input_json_delta", "partial_json": part} for part in (arguments[:12], arguments[12:])]
                    else:
                        start = {"type": "text", "text": ""}
                        deltas = [{"type": "text_delta", "text": block["text"]}]
                    events.append({"type": "content_block_start", "index": i, "content_block": start})
                    events.extend({"type": "content_block_delta", "index": i, "delta": delta} for delta in deltas)
                    events.append({"type": "content_block_stop", "index": i})
                events.extend([{"type": "message_delta", "delta": {"stop_reason": response["stop_reason"], "stop_sequence": None},
                                "usage": {"output_tokens": 20}}, {"type": "message_stop"}])
                data = "".join("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n" for event in events)
                return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"}, request=request)
        return httpx.Response(200, json=response, request=request)

    def decode_response(self, response, stream):
        self.assertEqual(response.status_code, 200, response.text)
        if not stream:
            return response.json()
        blocks, arguments, stopped = {}, {}, set()
        result = {"content": []}
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(events[-1]["type"], "message_stop", response.text)
        for event in events:
            kind = event["type"]
            if kind == "content_block_start":
                index = event["index"]
                self.assertNotIn(index, blocks)
                blocks[index] = event["content_block"]
                arguments[index] = ""
            elif kind == "content_block_delta":
                index, delta = event["index"], event["delta"]
                self.assertNotIn(index, stopped)
                self.assertIn(index, blocks)
                if delta["type"] == "input_json_delta":
                    arguments[index] += delta["partial_json"]
                else:
                    field = {"text_delta": "text", "thinking_delta": "thinking", "signature_delta": "signature"}[delta["type"]]
                    blocks[index][field] = blocks[index].get(field, "") + delta[field]
            elif kind == "content_block_stop":
                self.assertNotIn(event["index"], stopped)
                stopped.add(event["index"])
            elif kind == "message_delta":
                result.update(event["delta"])
        self.assertEqual(stopped, set(blocks))
        for index, block in blocks.items():
            if block["type"] == "tool_use":
                block["input"] = json.loads(arguments[index])
        result["content"] = list(blocks.values())
        return result

    def test_multiturn_tools_and_thinking_actual_wire(self):
        self.tool_fixture = True
        for style in ("anthropic", "openai"):
            for stream in (False, True):
                with self.subTest(style=style, stream=stream), patch.object(server, "MINIMAX_API_STYLE", style):
                    messages = [{"role": "user", "content": "Use echo twice"}]
                    payload = {"model": "MiniMax-M3.1-Flash", "max_tokens": 4096, "stream": stream,
                               "messages": messages, "tools": [{"name": "echo", "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}}}]}
                    first = self.decode_response(self.client.post("/v1/messages", json=payload), stream)
                    self.assertEqual(first["stop_reason"], "tool_use")
                    self.assertEqual(first["content"][0]["type"], "thinking")
                    self.assertEqual(first["content"][0]["thinking"], "I should use both tools.")
                    if style == "anthropic":
                        self.assertEqual(first["content"][0]["signature"], "opaque-signature-fixture")
                    calls = [block for block in first["content"] if block["type"] == "tool_use"]
                    self.assertEqual(len(calls), 2)
                    for i, call in enumerate(calls):
                        self.assertEqual(call, {"type": "tool_use", "id": f"call_fixture_{i}", "name": "echo",
                                                "input": {"text": f"value {i}", "nested": {"n": i}}})
                    results = [{"type": "tool_result", "tool_use_id": call["id"],
                                "content": [{"type": "text", "text": f"Done {i}"}], "is_error": False} for i, call in enumerate(calls)]
                    messages.extend([{"role": "assistant", "content": first["content"]}, {"role": "user", "content": results}])
                    second = self.decode_response(self.client.post("/v1/messages", json=payload), stream)
                    self.assertEqual(second["stop_reason"], "end_turn")
                    self.assertEqual(second["content"], [{"type": "text", "text": "Both done."}])
                    wire = self.calls[-1][1]
                    if style == "anthropic":
                        self.assertEqual(wire["messages"], messages)
                    else:
                        self.assertEqual([message["role"] for message in wire["messages"]], ["user", "assistant", "tool", "tool"])
                        assistant = wire["messages"][1]
                        self.assertEqual(assistant["reasoning_content"], "I should use both tools.")
                        self.assertEqual(len(assistant["tool_calls"]), 2)
                        for i, call in enumerate(assistant["tool_calls"]):
                            self.assertEqual(call["id"], calls[i]["id"])
                            self.assertEqual(call["function"]["name"], "echo")
                            self.assertEqual(json.loads(call["function"]["arguments"]), calls[i]["input"])
                            self.assertEqual(wire["messages"][i + 2]["tool_call_id"], calls[i]["id"])
                            self.assertEqual(wire["messages"][i + 2]["content"], results[i]["content"])

    def test_saved_tool_history_actual_wire(self):
        messages = [
            {"role": "user", "content": "Use echo"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tool_fixture", "name": "echo", "input": {"text": "hello"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_fixture", "content": "Done", "is_error": True},
                                         {"type": "text", "text": "Continue"}]},
        ]
        for style in ("anthropic", "openai"):
            for stream in (False, True):
                with self.subTest(style=style, stream=stream), patch.object(server, "MINIMAX_API_STYLE", style):
                    _, wire = self.send(messages=messages, stream=stream, tools=[{
                        "name": "echo", "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}}
                    }])
                    if style == "anthropic":
                        self.assertEqual(wire["messages"], messages)
                    else:
                        self.assertEqual([message["role"] for message in wire["messages"]], ["user", "assistant", "tool", "user"])
                        call = wire["messages"][1]["tool_calls"][0]
                        self.assertEqual(call["id"], "tool_fixture")
                        self.assertEqual(call["function"]["name"], "echo")
                        self.assertEqual(json.loads(call["function"]["arguments"]), {"text": "hello"})
                        self.assertEqual(wire["messages"][2], {"role": "tool", "tool_call_id": "tool_fixture", "content": "Done"})
                        self.assertEqual(wire["messages"][3]["content"], [{"type": "text", "text": "Continue"}])

    def test_saved_thinking_signature_actual_wire(self):
        messages = [
            {"role": "user", "content": "Think"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "Preserve me", "signature": "opaque-saved-signature"},
                                               {"type": "text", "text": "Ready"}]},
            {"role": "user", "content": "Continue"},
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                _, wire = self.send(messages=messages, stream=stream)
                self.assertEqual(wire["messages"], messages)

    def test_default_protocol_region_and_stream_matrix(self):
        for style in ("anthropic", "openai"):
            for host in ("api.minimax.io", "api.minimaxi.com"):
                for stream in (False, True):
                    for alias in ("claude-haiku", "claude-sonnet"):
                        with self.subTest(style=style, host=host, stream=stream, alias=alias), patch.multiple(
                            server, MINIMAX_API_STYLE=style,
                            MINIMAX_BASE_URL=f"https://{host}/v1",
                            MINIMAX_ANTHROPIC_BASE_URL=f"https://{host}/anthropic",
                        ):
                            request, body = self.send(alias, stream=stream)
                            self.assertEqual(request.url.host, host)
                            self.assertEqual(body["model"], "MiniMax-M3.1-Flash")
                            self.assertEqual(body["thinking"], {"type": "adaptive"})
                            self.assertEqual(body.get("max_tokens", body.get("max_completion_tokens")), 524288)
                            if style == "anthropic":
                                self.assertEqual(body["output_config"], {"effort": "max"})
                                self.assertNotIn("reasoning_effort", body)
                                self.assertEqual(request.headers["x-api-key"], "offline-test-key")
                            else:
                                self.assertEqual(body["reasoning_effort"], "max")
                                self.assertNotIn("output_config", body)
                                self.assertEqual(request.headers["authorization"], "Bearer offline-test-key")
                            self.assertNotIn("extra_body", body)
                            self.assertNotIn("service_tier", body)
                            self.assertNotIn("reasoning_split", body)

    def test_effort_and_legacy_enabled_thinking(self):
        for style in ("anthropic", "openai"):
            for effort in ("low", "medium", "high", "xhigh", "max"):
                for field in ("output_config", "reasoning_effort"):
                    with self.subTest(style=style, effort=effort, field=field), patch.object(server, "MINIMAX_API_STYLE", style):
                        options = {field: {"effort": effort} if field == "output_config" else effort}
                        _, body = self.send("MiniMax-M3.1-Flash", thinking={"type": "enabled", "budget_tokens": 10000}, **options)
                        self.assertEqual(body["thinking"], {"type": "adaptive"})
                        self.assertEqual(body.get("reasoning_effort", body.get("output_config", {}).get("effort")), effort)

    def test_flash_rejects_disabled_and_invalid_effort_before_transport(self):
        for options in ({"thinking": {"type": "disabled"}}, {"thinking": {"enabled": False}},
                        {"output_config": {"effort": "none"}}, {"reasoning_effort": "none"},
                        {"output_config": {"effort": "invalid"}}):
            with self.subTest(options=options):
                response = self.client.post("/v1/messages", json={
                    "model": "claude-haiku", "max_tokens": 4096,
                    "messages": [{"role": "user", "content": "Hello"}], **options,
                })
                self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.calls, [])

    def test_m3_thinking_and_effort(self):
        for style in ("anthropic", "openai"):
            for thinking in (None, "adaptive", "disabled"):
                with self.subTest(style=style, thinking=thinking), patch.object(server, "MINIMAX_API_STYLE", style):
                    options = {"thinking": {"type": thinking}} if thinking else {}
                    _, body = self.send("MiniMax-M3", output_config={"effort": "max"}, **options)
                    self.assertEqual(body["model"], "MiniMax-M3")
                    self.assertEqual(body.get("thinking"), options.get("thinking"))
                    self.assertNotIn("output_config", body)
                    self.assertNotIn("reasoning_effort", body)

    def test_explicit_models_and_environment_overrides(self):
        for style in ("anthropic", "openai"):
            for model in ("MiniMax-M3", "MiniMax-M2.7", "minimax/custom-sonnet", "MiniMax-custom"):
                with self.subTest(style=style, model=model), patch.multiple(server, MINIMAX_API_STYLE=style, PREFERRED_PROVIDER="anthropic"):
                    _, body = self.send(model)
                    expected = model.removeprefix("minimax/")
                    self.assertEqual(body["model"], expected)
                    token_request = server.TokenCountRequest(model=model, messages=[])
                    self.assertEqual(token_request.model, "minimax/" + expected)
        for prefix in ("", "minimax/"):
            with patch.dict(os.environ, {"BIG_MODEL": prefix + "MiniMax-M3", "SMALL_MODEL": prefix + "MiniMax-M2.7"}):
                importlib.reload(server)
                self.assertEqual(server.BIG_MODEL, prefix + "MiniMax-M3")
                self.assertEqual(server.SMALL_MODEL, prefix + "MiniMax-M2.7")
                for alias, expected in (("claude-sonnet", "MiniMax-M3"), ("claude-haiku", "MiniMax-M2.7")):
                    _, body = self.send(alias)
                    self.assertEqual(body["model"], expected)

    def test_token_count_alias_and_saved_model_routing(self):
        for style in ("anthropic", "openai"):
            for model, expected in (("claude-haiku", "MiniMax-M3.1-Flash"),
                                    ("MiniMax-M2.7", "MiniMax-M2.7"),
                                    ("minimax/custom-sonnet", "custom-sonnet")):
                with self.subTest(style=style, model=model), patch.object(server, "MINIMAX_API_STYLE", style), patch.object(server.litellm, "token_counter", return_value=12) as counter:
                    response = self.client.post("/v1/messages/count_tokens", json={
                        "model": model, "messages": [{"role": "user", "content": "Hello"}],
                    })
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json(), {"input_tokens": 12})
                    self.assertEqual(counter.call_args.kwargs["model"], style + "/" + expected)
                    self.assertEqual(counter.call_args.kwargs["api_base"], server.MINIMAX_BASE_URL if style == "openai" else server.MINIMAX_ANTHROPIC_BASE_URL)

    def test_non_minimax_defaults_and_image_conversion_unchanged(self):
        with patch.dict(os.environ, {"PREFERRED_PROVIDER": "openai"}):
            importlib.reload(server)
            self.assertEqual(server.BIG_MODEL, "gpt-4.1")
            self.assertEqual(server.SMALL_MODEL, "gpt-4.1-mini")
        self.assertEqual(server.MINIMAX_MODELS, ["MiniMax-M3.1-Flash", "MiniMax-M3"])
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}}
        request = server.MessagesRequest(model="minimax/MiniMax-M3.1-Flash", max_tokens=4096,
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}, image]}])
        converted = server.convert_anthropic_to_litellm(request)
        self.assertEqual(converted["messages"][0]["content"][1], image)


if __name__ == "__main__":
    unittest.main()
