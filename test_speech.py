import json
import unittest
from unittest.mock import patch

from fastapi import HTTPException

import server


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


class FakeStreamResponse(FakeResponse):
    async def aiter_lines(self):
        yield 'data: {"data":{"audio":"4944","status":1},"base_resp":{"status_code":0}}'
        yield 'data: {"data":{"audio":"33","status":2},"base_resp":{"status_code":0}}'
        yield "data: [DONE]"


class FakeStreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeAsyncClient:
    response = None
    last_request = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url, headers, json):
        type(self).last_request = (url, headers, json)
        return type(self).response

    def stream(self, method, url, headers, json):
        type(self).last_request = (url, headers, json)
        return FakeStreamContext(type(self).response)


class SpeechRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.success_payload = {
            "data": {"audio": "494433", "status": 2},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        FakeAsyncClient.response = FakeResponse(self.success_payload)
        FakeAsyncClient.last_request = None

    async def test_decodes_hex_audio_and_forwards_supported_fields(self):
        request = server.SpeechRequest(
            model="speech-2.8-hd",
            text="Hello",
            language_boost="English",
            voice_setting={"voice_id": "English_expressive_narrator"},
            pronunciation_dict={"tone": ["Hello/Hello"]},
            audio_setting={"format": "mp3", "sample_rate": 32000},
            voice_modify={"pitch": 1},
            subtitle_enable=True,
        )
        with (
            patch.object(server, "MINIMAX_API_KEY", "test-key"),
            patch.object(server, "httpx") as mock_httpx,
        ):
            mock_httpx.AsyncClient = FakeAsyncClient
            response = await server.create_speech(request)

        self.assertEqual(response.body, b"ID3")
        self.assertEqual(response.media_type, "audio/mpeg")
        url, headers, payload = FakeAsyncClient.last_request
        self.assertEqual(url, "https://api.minimax.io/v1/t2a_v2")
        self.assertEqual(headers, {"Authorization": "Bearer test-key"})
        self.assertEqual(payload["model"], "speech-2.8-hd")
        self.assertEqual(
            payload["voice_setting"]["voice_id"], "English_expressive_narrator"
        )
        self.assertEqual(payload["audio_setting"]["format"], "mp3")
        self.assertTrue(payload["subtitle_enable"])

    async def test_uses_configured_regional_endpoint(self):
        request = server.SpeechRequest(text="Hello")
        with (
            patch.object(server, "MINIMAX_API_KEY", "test-key"),
            patch.object(
                server,
                "MINIMAX_SPEECH_URL",
                "https://api.minimaxi.com/v1/t2a_v2",
            ),
            patch.object(server.httpx, "AsyncClient", FakeAsyncClient),
        ):
            await server.create_speech(request)

        url, _, _ = FakeAsyncClient.last_request
        self.assertEqual(url, "https://api.minimaxi.com/v1/t2a_v2")

    async def test_url_output_preserves_response_json(self):
        url_payload = {
            "data": {"audio": "https://example.invalid/speech.mp3", "status": 2},
            "base_resp": {"status_code": 0},
        }
        FakeAsyncClient.response = FakeResponse(url_payload)
        request = server.SpeechRequest(text="Hello", output_format="url")
        with (
            patch.object(server, "MINIMAX_API_KEY", "test-key"),
            patch.object(server.httpx, "AsyncClient", FakeAsyncClient),
        ):
            response = await server.create_speech(request)

        self.assertEqual(json.loads(response.body), url_payload)

    async def test_stream_decodes_audio_chunks(self):
        FakeAsyncClient.response = FakeStreamResponse({})
        request = server.SpeechRequest(text="Hello", stream=True)
        with (
            patch.object(server, "MINIMAX_API_KEY", "test-key"),
            patch.object(server.httpx, "AsyncClient", FakeAsyncClient),
        ):
            response = await server.create_speech(request)
            chunks = [chunk async for chunk in response.body_iterator]

        self.assertEqual(chunks, [b"ID", b"3"])

    async def test_rejects_url_output_for_streaming(self):
        request = server.SpeechRequest(text="Hello", stream=True, output_format="url")
        with patch.object(server, "MINIMAX_API_KEY", "test-key"):
            with self.assertRaises(HTTPException) as context:
                await server.create_speech(request)
        self.assertEqual(context.exception.status_code, 400)

    async def test_surfaces_api_error(self):
        FakeAsyncClient.response = FakeResponse(
            {
                "data": None,
                "base_resp": {"status_code": 1001, "status_msg": "Invalid request"},
            }
        )
        request = server.SpeechRequest(text="Hello")
        with (
            patch.object(server, "MINIMAX_API_KEY", "test-key"),
            patch.object(server.httpx, "AsyncClient", FakeAsyncClient),
        ):
            with self.assertRaises(HTTPException) as context:
                await server.create_speech(request)
        self.assertEqual(context.exception.status_code, 502)
        self.assertEqual(context.exception.detail, "Invalid request")


if __name__ == "__main__":
    unittest.main()
