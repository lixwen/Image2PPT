from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
DECK_DIR = ROOT / "scripts" / "deck"
if str(DECK_DIR) not in sys.path:
    sys.path.insert(0, str(DECK_DIR))

from calibrate_text_sizes import (  # noqa: E402
    _backend_capped_ratio,
    _calibrated_box_width,
)


class BaiduCalibrationLimitTests(unittest.TestCase):
    def test_baidu_size_calibration_does_not_enlarge_text(self) -> None:
        self.assertEqual(
            _backend_capped_ratio({"ocr_backend": "baidu"}, 1.35),
            1.0,
        )
        self.assertEqual(
            _backend_capped_ratio({"ocr_backend": "baidu"}, 0.82),
            0.82,
        )

    def test_non_baidu_size_calibration_keeps_existing_behavior(self) -> None:
        self.assertEqual(
            _backend_capped_ratio({"ocr_backend": "paddle"}, 1.35),
            1.35,
        )

    def test_baidu_box_width_ignores_render_overflow_expansion(self) -> None:
        self.assertEqual(
            _calibrated_box_width(
                {"ocr_backend": "baidu"},
                current_w=120,
                target_w=100,
                rendered_w=260,
                scale_after=1.0,
            ),
            120,
        )

    def test_non_baidu_box_width_keeps_render_overflow_expansion(self) -> None:
        self.assertEqual(
            _calibrated_box_width(
                {"ocr_backend": "paddle"},
                current_w=120,
                target_w=100,
                rendered_w=260,
                scale_after=1.0,
            ),
            272,
        )


if __name__ == "__main__":
    unittest.main()
