from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
OCR_DIR = ROOT / "scripts" / "ocr"
if str(OCR_DIR) not in sys.path:
    sys.path.insert(0, str(OCR_DIR))

from ocr_baidu import (  # noqa: E402
    BAIDU_ACCURATE_ENDPOINT,
    BAIDU_GENERAL_ENDPOINT,
    BaiduOcrApiError,
    BaiduOcrClient,
    BaiduOcrConfig,
)


class FakeBaiduClient(BaiduOcrClient):
    def __init__(self, config: BaiduOcrConfig, responses: list[dict]):
        super().__init__(config)
        self.responses = list(responses)
        self.urls: list[str] = []

    def access_token(self) -> str:
        return "fake-token"

    def _post_json(self, url: str, *, data: bytes, headers: dict[str, str]) -> dict:
        self.urls.append(url)
        if not self.responses:
            self.fail("No fake Baidu response queued")
        return self.responses.pop(0)

    def fail(self, message: str) -> None:
        raise AssertionError(message)


def success_payload() -> dict:
    return {
        "words_result": [
            {
                "words": "测试",
                "location": {"left": 10, "top": 20, "width": 40, "height": 18},
                "probability": {"average": 0.99},
                "chars": [
                    {
                        "char": "测",
                        "char_prob": 0.98,
                        "location": {"left": 10, "top": 20, "width": 20, "height": 18},
                    },
                    {
                        "char": "试",
                        "char_prob": 0.98,
                        "location": {"left": 30, "top": 20, "width": 20, "height": 18},
                    },
                ],
            }
        ]
    }


class BaiduOcrProviderPoolTests(unittest.TestCase):
    def test_default_provider_order_is_high_then_standard(self) -> None:
        config = BaiduOcrConfig(api_key="", secret_key="", access_token="token")
        providers = config.providers()

        self.assertEqual([p.name for p in providers], ["high", "standard"])
        self.assertEqual(providers[0].endpoint, BAIDU_ACCURATE_ENDPOINT)
        self.assertEqual(providers[1].endpoint, BAIDU_GENERAL_ENDPOINT)

    def test_quota_exhausted_falls_back_to_standard(self) -> None:
        config = BaiduOcrConfig(
            api_key="",
            secret_key="",
            access_token="token",
            retries=0,
            provider_order="high,standard",
        )
        client = FakeBaiduClient(
            config,
            [
                {
                    "error_code": 17,
                    "error_msg": "Open api daily request limit reached",
                },
                success_payload(),
            ],
        )
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            tmp.write(b"not-a-real-png")
            tmp.flush()
            items = client.recognize_image(Path(tmp.name), 0.3)

        self.assertEqual(client.last_provider, "standard")
        self.assertEqual(items[0]["baidu_provider"], "standard")
        self.assertIn("/ocr/v1/accurate", client.urls[0])
        self.assertIn("/ocr/v1/general", client.urls[1])

    def test_qps_limit_does_not_fall_back_to_standard(self) -> None:
        config = BaiduOcrConfig(
            api_key="",
            secret_key="",
            access_token="token",
            retries=0,
            provider_order="high,standard",
        )
        client = FakeBaiduClient(
            config,
            [
                {
                    "error_code": 18,
                    "error_msg": "Open api qps request limit reached",
                },
            ],
        )
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            tmp.write(b"not-a-real-png")
            tmp.flush()
            with self.assertRaises(BaiduOcrApiError):
                client.recognize_image(Path(tmp.name), 0.3)

        self.assertEqual(len(client.urls), 1)
        self.assertIn("/ocr/v1/accurate", client.urls[0])


if __name__ == "__main__":
    unittest.main()
