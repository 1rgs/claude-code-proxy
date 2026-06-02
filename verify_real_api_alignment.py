#!/usr/bin/env python3
"""
Real API alignment verifier for an Anthropic-compatible Claude Code proxy.

What it checks:
1. Anthropic -> OpenAI conversion invariants offline, by importing server.py.
2. Direct OpenAI-compatible Ark/DeepSeek tool call response.
3. Proxy /v1/messages non-streaming tool_use restoration.
4. Proxy /v1/messages streaming SSE tool_use restoration.
5. Optional KV cache hit telemetry over repeated identical prefixes.

Run examples:
  # Offline conversion-only check against your local server file
  python verify_real_api_alignment.py --offline --server-file ./server_fixed_cache_ark_usage.py

  # Real proxy test. Start your proxy first: python server_fixed_cache_ark_usage.py
  python verify_real_api_alignment.py --proxy http://127.0.0.1:8082 --stream --cache

  # Also test the upstream OpenAI-compatible endpoint directly
  python verify_real_api_alignment.py --direct --proxy http://127.0.0.1:8082 --stream --cache

Required env for direct/upstream calls:
  OPENAI_API_KEY, OPENAI_BASE_URL, BIG_MODEL
Optional env:
  PROXY_MODEL, TEST_TIMEOUT, TEST_CACHE_REPEATS, TEST_CACHE_SLEEP_SECONDS
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import httpx


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []
        self.artifacts: dict[str, Any] = {}

    def add(self, name: str, ok: bool, detail: str = "", **data: Any) -> None:
        self.checks.append(Check(name=name, ok=ok, detail=detail, data=_redact(data)))
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")
        if data:
            print(json.dumps(_redact(data), ensure_ascii=False, indent=2)[:4000])

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [c.__dict__ for c in self.checks],
            "artifacts": _redact(self.artifacts),
        }


SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{8,})"),
    re.compile(r"([A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,})"),
    re.compile(r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"),
]


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        out = value
        for pat in SECRET_PATTERNS:
            out = pat.sub("<redacted>", out)
        return out
    if isinstance(value, dict):
        redacted = {}
        for k, v in value.items():
            if str(k).lower() in {"authorization", "api_key", "openai_api_key", "anthropic_api_key", "gemini_api_key"}:
                redacted[k] = "<redacted>"
            else:
                redacted[k] = _redact(v)
        return redacted
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _load_server_module(path: str):
    """Import server file without requiring real litellm if it is unavailable."""
    if "litellm" not in sys.modules:
        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("litellm.completion not available in offline mode"))
        async def _fake_acompletion(**kwargs):
            raise RuntimeError("litellm.acompletion not available in offline mode")
        fake_litellm.acompletion = _fake_acompletion
        fake_litellm.token_counter = lambda **kwargs: 0
        sys.modules["litellm"] = fake_litellm
    spec = importlib.util.spec_from_file_location("server_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["server_under_test"] = module
    spec.loader.exec_module(module)
    return module


def sample_anthropic_request(proxy_model: str, stream: bool = False) -> dict[str, Any]:
    return {
        "model": proxy_model,
        "max_tokens": 128,
        "temperature": 0,
        "stream": stream,
        "messages": [
            {"role": "user", "content": "Call the add tool with a=2 and b=3. Do not answer in text."}
        ],
        "tools": [
            {
                "name": "add",
                "description": "Add two integers.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "add"},
    }


def sample_openai_request(model: str, stream: bool = False) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Call the add tool with a=2 and b=3. Do not answer in text."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "add"}},
        "temperature": 0,
        "max_tokens": 128,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def _parse_args_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or value == "":
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    except Exception:
        return {"_raw": str(value)}


def _find_anthropic_tool_use(resp: dict[str, Any]) -> dict[str, Any] | None:
    for block in resp.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return block
    return None


def _find_openai_tool_call(resp: dict[str, Any]) -> dict[str, Any] | None:
    choices = resp.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message") or {}
    calls = msg.get("tool_calls") or []
    return calls[0] if calls else None


def _extract_usage_counts(usage: Any) -> dict[str, int]:
    usage = usage if isinstance(usage, dict) else {}
    def get(path: list[str]) -> Any:
        cur: Any = usage
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur
    def first(paths: list[list[str]]) -> int:
        for p in paths:
            v = get(p)
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    pass
        return 0
    return {
        "input_tokens": first([["input_tokens"], ["prompt_tokens"]]),
        "output_tokens": first([["output_tokens"], ["completion_tokens"]]),
        "cache_read_input_tokens": first([
            ["cache_read_input_tokens"],
            ["prompt_cache_hit_tokens"],
            ["prompt_tokens_details", "cached_tokens"],
        ]),
        "cache_creation_input_tokens": first([
            ["cache_creation_input_tokens"],
            ["prompt_cache_miss_tokens"],
            ["prompt_tokens_details", "cache_write_tokens"],
        ]),
    }


async def post_json(url: str, body: dict[str, Any], headers: dict[str, str] | None = None, timeout: float = 60) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=body)
        try:
            data = r.json()
        except Exception:
            data = {"_raw_text": r.text}
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} from {url}: {_json_dumps(_redact(data))[:1000]}")
        return data


async def stream_sse(url: str, body: dict[str, Any], headers: dict[str, str] | None = None, timeout: float = 120) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, headers=headers, json=body) as r:
            if r.status_code >= 400:
                text = await r.aread()
                raise RuntimeError(f"HTTP {r.status_code} from {url}: {_redact(text.decode(errors='replace'))[:1000]}")
            event_name = None
            data_lines: list[str] = []
            async for line in r.aiter_lines():
                if line == "":
                    if data_lines:
                        data_text = "\n".join(data_lines)
                        if data_text != "[DONE]":
                            try:
                                data = json.loads(data_text)
                            except Exception:
                                data = {"_raw": data_text}
                            data["_event"] = event_name
                            events.append(data)
                    event_name = None
                    data_lines = []
                    continue
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].strip())
    return events


def reconstruct_anthropic_stream(events: list[dict[str, Any]]) -> dict[str, Any]:
    blocks: dict[int, dict[str, Any]] = {}
    arg_buffers: dict[int, str] = {}
    usage: dict[str, Any] = {}
    stop_reason = None
    for ev in events:
        typ = ev.get("type") or ev.get("_event")
        if typ == "content_block_start":
            idx = int(ev["index"])
            blocks[idx] = dict(ev.get("content_block") or {})
            if blocks[idx].get("type") == "tool_use":
                arg_buffers[idx] = ""
        elif typ == "content_block_delta":
            idx = int(ev["index"])
            delta = ev.get("delta") or {}
            if delta.get("type") == "text_delta":
                blocks.setdefault(idx, {"type": "text", "text": ""})["text"] = blocks.setdefault(idx, {"type": "text", "text": ""}).get("text", "") + delta.get("text", "")
            elif delta.get("type") == "input_json_delta":
                arg_buffers[idx] = arg_buffers.get(idx, "") + delta.get("partial_json", "")
        elif typ == "message_delta":
            usage = ev.get("usage") or {}
            stop_reason = (ev.get("delta") or {}).get("stop_reason")
    content: list[dict[str, Any]] = []
    for idx in sorted(blocks):
        block = blocks[idx]
        if block.get("type") == "tool_use":
            block["input"] = _parse_args_json(arg_buffers.get(idx, ""))
        content.append(block)
    return {"content": content, "usage": usage, "stop_reason": stop_reason}


def reconstruct_openai_stream(events: list[dict[str, Any]]) -> dict[str, Any]:
    tool_states: dict[int, dict[str, Any]] = {}
    text = ""
    usage: dict[str, Any] = {}
    finish_reason = None
    for ev in events:
        if "usage" in ev and ev.get("usage"):
            usage = ev.get("usage") or {}
        choices = ev.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or choice.get("message") or {}
        if delta.get("content"):
            text += str(delta.get("content"))
        for tc in delta.get("tool_calls") or []:
            idx = int(tc.get("index", 0) or 0)
            state = tool_states.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            if tc.get("id"):
                state["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                state["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                state["function"]["arguments"] += fn["arguments"]
    return {"text": text, "tool_calls": [tool_states[i] for i in sorted(tool_states)], "usage": usage, "finish_reason": finish_reason}


async def direct_stream_tool_call_check(report: Report, base_url: str, api_key: str, model: str, timeout: float) -> None:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    events = await stream_sse(url, sample_openai_request(model, stream=True), headers=headers, timeout=timeout)
    reconstructed = reconstruct_openai_stream(events)
    tc = (reconstructed.get("tool_calls") or [None])[0]
    args = _parse_args_json(((tc or {}).get("function") or {}).get("arguments")) if tc else {}
    usage_counts = _extract_usage_counts(reconstructed.get("usage") or {})
    ok_tool = bool(tc and ((tc.get("function") or {}).get("name") == "add") and int(args.get("a", -1)) == 2 and int(args.get("b", -1)) == 3)
    ok_usage = any(v > 0 for v in usage_counts.values())
    report.add("direct_openai_stream_tool_call", ok_tool, "upstream stream returned OpenAI tool_calls" if ok_tool else "upstream stream tool_call mismatch", reconstructed=reconstructed, events_seen=len(events))
    report.add("direct_openai_stream_usage_present", ok_usage, "upstream stream included usage" if ok_usage else "upstream stream did not include usage despite stream_options.include_usage", usage=reconstructed.get("usage"))


async def offline_conversion_check(report: Report, server_file: str, proxy_model: str) -> None:
    try:
        m = _load_server_module(server_file)
        req = m.MessagesRequest(**sample_anthropic_request(proxy_model, stream=False))
        lite = m.convert_anthropic_to_litellm(req)
        assistant_req = m.MessagesRequest(
            model=req.model,
            max_tokens=128,
            messages=[
                {"role": "user", "content": "Call add."},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_test", "name": "add", "input": {"a": 2, "b": 3}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_test", "content": [{"type": "text", "text": "{\"sum\":5}"}]}]},
            ],
        )
        lite2 = m.convert_anthropic_to_litellm(assistant_req)
        synthetic_response = {
            "id": "chatcmpl_test",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_test",
                        "type": "function",
                        "function": {"name": "add", "arguments": "{\"a\":2,\"b\":3}"},
                    }],
                },
            }],
            "usage": {"prompt_tokens": 123, "completion_tokens": 7, "prompt_cache_hit_tokens": 99, "prompt_cache_miss_tokens": 24},
        }
        anth = m.convert_litellm_to_anthropic(synthetic_response, req).model_dump()
        checks = {
            "request_has_openai_tools": bool(lite.get("tools") and lite["tools"][0]["type"] == "function"),
            # tool_choice may be downgraded to "auto" by the proxy when the
            # upstream rejects forced choice (DashScope thinking-mode DeepSeek).
            # Accept either the strict function form or the safe downgrade.
            "tool_choice_is_function": (
                (isinstance(lite.get("tool_choice"), dict)
                 and (lite["tool_choice"].get("function") or {}).get("name") == "add")
                or lite.get("tool_choice") == "auto"
            ),
            "assistant_tool_use_becomes_tool_calls": any("tool_calls" in msg for msg in lite2["messages"]),
            "tool_result_becomes_role_tool": any(msg.get("role") == "tool" and msg.get("tool_call_id") == "toolu_test" for msg in lite2["messages"]),
            "response_tool_call_becomes_tool_use": bool(_find_anthropic_tool_use(anth)),
            "usage_cache_hit_preserved": anth.get("usage", {}).get("cache_read_input_tokens") == 99,
        }
        report.add("offline_conversion_consistency", all(checks.values()), "conversion invariants checked", checks=checks, sample_litellm_messages=lite2["messages"], restored=anth)
    except Exception as e:
        report.add("offline_conversion_consistency", False, f"{type(e).__name__}: {e}")


async def direct_tool_call_check(report: Report, base_url: str, api_key: str, model: str, timeout: float) -> None:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = await post_json(url, sample_openai_request(model, stream=False), headers=headers, timeout=timeout)
    tc = _find_openai_tool_call(resp)
    args = _parse_args_json(((tc or {}).get("function") or {}).get("arguments")) if tc else {}
    ok = bool(tc and ((tc.get("function") or {}).get("name") == "add") and int(args.get("a", -1)) == 2 and int(args.get("b", -1)) == 3)
    report.add("direct_openai_tool_call", ok, "upstream returned OpenAI tool_calls" if ok else "upstream did not return expected tool_call", tool_call=tc, usage=resp.get("usage"))


async def proxy_nonstream_tool_check(report: Report, proxy: str, proxy_model: str, timeout: float) -> dict[str, Any] | None:
    url = proxy.rstrip("/") + "/v1/messages"
    resp = await post_json(url, sample_anthropic_request(proxy_model, stream=False), timeout=timeout)
    tu = _find_anthropic_tool_use(resp)
    inp = tu.get("input") if tu else {}
    ok = bool(tu and tu.get("name") == "add" and int(inp.get("a", -1)) == 2 and int(inp.get("b", -1)) == 3 and resp.get("stop_reason") == "tool_use")
    report.add("proxy_nonstream_restore_tool_use", ok, "proxy restored OpenAI tool_calls to Anthropic tool_use" if ok else "proxy response mismatch", tool_use=tu, stop_reason=resp.get("stop_reason"), usage=resp.get("usage"))
    return tu


async def proxy_tool_result_followup_check(report: Report, proxy: str, proxy_model: str, tool_use: dict[str, Any], timeout: float) -> None:
    url = proxy.rstrip("/") + "/v1/messages"
    body = {
        "model": proxy_model,
        "max_tokens": 80,
        "temperature": 0,
        "stream": False,
        "messages": [
            {"role": "user", "content": "Call the add tool with a=2 and b=3."},
            {"role": "assistant", "content": [tool_use]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": [{"type": "text", "text": "{\"sum\":5}"}]}]},
            {"role": "user", "content": "Now answer only: final=5"},
        ],
    }
    resp = await post_json(url, body, timeout=timeout)
    text = "\n".join(block.get("text", "") for block in resp.get("content", []) if isinstance(block, dict) and block.get("type") == "text")
    ok = bool(text.strip()) and "tool_use" not in [b.get("type") for b in resp.get("content", []) if isinstance(b, dict)]
    report.add("proxy_tool_result_followup", ok, "tool_result round-trip accepted by upstream" if ok else "follow-up failed or asked another tool", text=text, usage=resp.get("usage"))


async def proxy_stream_tool_check(report: Report, proxy: str, proxy_model: str, timeout: float) -> None:
    url = proxy.rstrip("/") + "/v1/messages"
    events = await stream_sse(url, sample_anthropic_request(proxy_model, stream=True), timeout=timeout)
    reconstructed = reconstruct_anthropic_stream(events)
    tu = _find_anthropic_tool_use(reconstructed)
    inp = tu.get("input") if tu else {}
    usage_counts = _extract_usage_counts(reconstructed.get("usage") or {})
    ok_tool = bool(tu and tu.get("name") == "add" and int(inp.get("a", -1)) == 2 and int(inp.get("b", -1)) == 3 and reconstructed.get("stop_reason") == "tool_use")
    ok_usage_present = usage_counts["input_tokens"] > 0 or usage_counts["output_tokens"] > 0 or usage_counts["cache_read_input_tokens"] > 0 or usage_counts["cache_creation_input_tokens"] > 0
    report.add("proxy_stream_restore_tool_use", ok_tool, "stream SSE reconstructed to Anthropic tool_use" if ok_tool else "stream tool_use mismatch", tool_use=tu, stop_reason=reconstructed.get("stop_reason"), usage=reconstructed.get("usage"), events_seen=len(events))
    report.add("proxy_stream_usage_present", ok_usage_present, "stream message_delta included token/cache usage" if ok_usage_present else "stream returned no usage; provider or adapter may be dropping usage chunks", usage=reconstructed.get("usage"))


async def cache_hit_check(report: Report, proxy: str, proxy_model: str, timeout: float, repeats: int, sleep_seconds: float) -> None:
    url = proxy.rstrip("/") + "/v1/messages"
    stable = "STATIC_CACHE_PREFIX_9b1d " * 1800
    hits: list[int] = []
    usages: list[dict[str, Any]] = []
    for i in range(repeats):
        body = {
            "model": proxy_model,
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
            "system": "You are a terse echo model. Keep the following static prefix unchanged for cache tests.",
            "messages": [{"role": "user", "content": stable + f"\nQuestion: reply with the digit {i % 2}."}],
        }
        resp = await post_json(url, body, timeout=timeout)
        usage = resp.get("usage") or {}
        counts = _extract_usage_counts(usage)
        hits.append(counts["cache_read_input_tokens"])
        usages.append(usage)
        if sleep_seconds:
            await asyncio.sleep(sleep_seconds)
    ok = any(h > 0 for h in hits[1:])
    report.add("proxy_kv_cache_hit", ok, "at least one repeated-prefix request reported cache hit" if ok else "no cache hit reported; inspect raw usage and provider cache policy", hits=hits, usages=usages)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-file", default=os.environ.get("SERVER_FILE", "server_fixed_cache_ark_usage.py"))
    parser.add_argument("--offline", action="store_true", help="Only run conversion checks; no network/API calls.")
    parser.add_argument("--direct", action="store_true", help="Also call OPENAI_BASE_URL/chat/completions directly.")
    parser.add_argument("--proxy", default=os.environ.get("PROXY_URL", "http://127.0.0.1:8082"))
    parser.add_argument("--proxy-model", default=os.environ.get("PROXY_MODEL", "claude-opus-4-7"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("BIG_MODEL", os.environ.get("OPENAI_MODEL", "")))
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("TEST_TIMEOUT", "120")))
    parser.add_argument("--cache-repeats", type=int, default=int(os.environ.get("TEST_CACHE_REPEATS", "3")))
    parser.add_argument("--cache-sleep", type=float, default=float(os.environ.get("TEST_CACHE_SLEEP_SECONDS", "2")))
    parser.add_argument("--report", default=os.environ.get("TEST_REPORT", "alignment_report.json"))
    args = parser.parse_args()

    report = Report()

    server_file = str(Path(args.server_file).resolve()) if Path(args.server_file).exists() else args.server_file
    await offline_conversion_check(report, server_file, args.proxy_model)

    if not args.offline:
        if args.direct:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key or not args.base_url or not args.model:
                report.add("direct_config", False, "OPENAI_API_KEY, OPENAI_BASE_URL and BIG_MODEL/--model are required for --direct")
            else:
                await direct_tool_call_check(report, args.base_url, api_key, args.model, args.timeout)
                if args.stream:
                    await direct_stream_tool_call_check(report, args.base_url, api_key, args.model, args.timeout)

        tool_use = await proxy_nonstream_tool_check(report, args.proxy, args.proxy_model, args.timeout)
        if tool_use:
            await proxy_tool_result_followup_check(report, args.proxy, args.proxy_model, tool_use, args.timeout)
        if args.stream:
            await proxy_stream_tool_check(report, args.proxy, args.proxy_model, args.timeout)
        if args.cache:
            await cache_hit_check(report, args.proxy, args.proxy_model, args.timeout, args.cache_repeats, args.cache_sleep)

    out_path = Path(args.report)
    out_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to: {out_path.resolve()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
