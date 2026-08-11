import unittest

from server import (
    Message,
    convert_multimodal_content_blocks,
    is_supported_multimodal_content,
)


class MultimodalContentTests(unittest.TestCase):
    def test_request_schema_accepts_video_content(self):
        message = Message(
            role="user",
            content=[
                {
                    "type": "video",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/sample.mp4",
                    },
                },
                {
                    "type": "text",
                    "text": "Describe this clip.",
                },
            ],
        )

        self.assertEqual(message.content[0].type, "video")

    def test_media_content_is_preserved_as_url_parts(self):
        converted = convert_multimodal_content_blocks(
            [
                {"type": "text", "text": "Compare these inputs."},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "aW1hZ2U=",
                    },
                },
                {
                    "type": "video",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/sample.mp4",
                    },
                },
            ]
        )

        self.assertEqual(
            converted,
            [
                {"type": "text", "text": "Compare these inputs."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                },
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/sample.mp4"},
                },
            ],
        )
        self.assertTrue(is_supported_multimodal_content(converted))


if __name__ == "__main__":
    unittest.main()
