#!/usr/bin/env python
"""Baidu OCR wrapper -> DeckWeaver OCR JSON.

Implements Baidu "通用文字识别（高精度含位置版）" and
"通用文字识别（标准含位置版）" as documented at:
https://cloud.baidu.com/doc/OCR/s/tk3h7y2aq
https://cloud.baidu.com/doc/OCR/s/vk3h7y58v

The module emits the same JSON shape as ``ocr_paddle.py`` so the rest of
the pipeline can stay backend-neutral:

    {"text": str, "x1": int, "y1": int, "x2": int, "y2": int,
     "confidence": float, "chars": [...], "char_boxes": [...]}
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


BAIDU_ACCURATE_ENDPOINT = (
    "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate"
)
BAIDU_GENERAL_ENDPOINT = (
    "https://aip.baidubce.com/rest/2.0/ocr/v1/general"
)
BAIDU_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
_WATERMARK_RE = re.compile(r"^(?:\d{3,4}\s*)?[0-9a-fA-F]{6,12}$")
_URL_RE = re.compile(r"https?:/+(?:www\.)?", re.IGNORECASE)
_PROVIDER_QUOTA_CODES = {17, 19, 216604}
_PROVIDER_UNAVAILABLE_CODES = {6, 216102}
_RETRYABLE_CODES = {1, 2, 4, 18, 216401, 216402, 216630}


class BaiduOcrError(RuntimeError):
    """Raised for user-actionable Baidu OCR failures."""


class BaiduOcrApiError(BaiduOcrError):
    """Baidu JSON error response with parsed error_code."""

    def __init__(self, code: Any, message: Any, provider: str):
        self.error_code = _as_int(code)
        self.error_msg = str(message or "")
        self.provider = provider
        code_text = code if self.error_code is None else self.error_code
        super().__init__(
            f"Baidu OCR provider '{provider}' API error "
            f"{code_text}: {self.error_msg}"
        )


@dataclass(frozen=True)
class BaiduOcrProvider:
    name: str
    endpoint: str


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _env_int(*names: str, default: int) -> int:
    value = _env(*names)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(*names: str, default: float) -> float:
    value = _env(*names)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
            return default


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _env_bool(*names: str, default: bool) -> bool:
    value = _env(*names)
    if not value:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _normalize_conf(value: Any, default: float = 1.0) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return default
    if conf > 1.0 and conf <= 100.0:
        conf /= 100.0
    return max(0.0, min(1.0, conf))


def _bbox_from_location(loc: dict[str, Any] | None) -> tuple[int, int, int, int]:
    loc = loc or {}
    left = int(round(float(loc.get("left", 0) or 0)))
    top = int(round(float(loc.get("top", 0) or 0)))
    width = int(round(float(loc.get("width", 0) or 0)))
    height = int(round(float(loc.get("height", 0) or 0)))
    return left, top, left + max(0, width), top + max(0, height)


def _bbox_from_points(points: list[dict[str, Any]] | None) -> tuple[int, int, int, int] | None:
    if not points:
        return None
    try:
        xs = [int(round(float(p["x"]))) for p in points]
        ys = [int(round(float(p["y"]))) for p in points]
    except (KeyError, TypeError, ValueError):
        return None
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _is_cjk(ch: str) -> bool:
    return "\u3400" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff"


def _normalize_url_slashes(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        value = match.group(0)
        scheme = "https://" if value.lower().startswith("https") else "http://"
        return scheme + ("www." if "www." in value.lower() else "")

    return _URL_RE.sub(repl, text)


def _normalize_symbols(text: str) -> str:
    """Fix Baidu OCR confusions that are common on PPT screenshots."""
    if not text:
        return text
    chars = list(_normalize_url_slashes(text))
    for idx, ch in enumerate(chars):
        if ch not in {"x", "X"}:
            continue
        prev_ch = chars[idx - 1] if idx > 0 else ""
        next_ch = chars[idx + 1] if idx + 1 < len(chars) else ""
        if _is_cjk(prev_ch) and _is_cjk(next_ch):
            chars[idx] = "×"
    return "".join(chars)


def _char_width_unit(ch: str) -> float:
    if ch.isspace():
        return 0.35
    if ch.isascii() and ch.isalnum():
        return 0.56
    if ch in ".,:;!|'`":
        return 0.25
    if ch in "()[]{}（）":
        return 0.34
    if ch in "-+/=":
        return 0.45
    if ch in "→←•·‧×":
        return 0.70
    return 1.0


def _split_box_by_units(text: str, box: list[int]) -> list[list[int]]:
    x1, y1, x2, y2 = (int(v) for v in box)
    units = [_char_width_unit(ch) for ch in text]
    total = sum(units)
    if total <= 0 or x2 <= x1 or y2 <= y1:
        return []
    out: list[list[int]] = []
    cur = 0.0
    width = float(x2 - x1)
    for idx, unit in enumerate(units):
        start = cur
        cur += unit
        px1 = int(round(x1 + (start / total) * width))
        px2 = int(round(x1 + (cur / total) * width))
        if idx == 0:
            px1 = x1
        if idx == len(text) - 1:
            px2 = x2
        if px2 <= px1:
            px2 = min(x2, px1 + 1)
        out.append([px1, y1, px2, y2])
    return out


def _align_char_boxes(
    text: str,
    char_texts: list[str],
    char_boxes: list[list[int]],
    fallback_box: list[int],
) -> tuple[list[str], list[list[int]]] | None:
    """Return one char box per text char, inserting boxes for spaces.

    Baidu's `chars` often omits whitespace. The downstream layout code is
    much more stable when `len(char_boxes) == len(text)`, so preserve real
    Baidu boxes where possible and synthesize only the missing whitespace
    or, as a last resort, split the whole line bbox by width units.
    """
    if not text:
        return None
    if len(char_texts) == len(char_boxes) and "".join(char_texts) == text:
        return list(char_texts), [list(b) for b in char_boxes]

    compact = "".join(ch for ch in text if not ch.isspace())
    if (char_texts and len(char_texts) == len(char_boxes)
            and "".join(char_texts) == compact):
        out_chars: list[str] = []
        out_boxes: list[list[int]] = []
        src_idx = 0
        last_box: list[int] | None = None
        for idx, ch in enumerate(text):
            if ch.isspace():
                next_box = char_boxes[src_idx] if src_idx < len(char_boxes) else None
                if last_box is not None and next_box is not None:
                    sx1 = int(last_box[2])
                    sx2 = int(next_box[0])
                    y1 = min(int(last_box[1]), int(next_box[1]))
                    y2 = max(int(last_box[3]), int(next_box[3]))
                    if sx2 <= sx1:
                        sx1 = sx2 = int(round((int(last_box[2]) + int(next_box[0])) / 2))
                    out_boxes.append([sx1, y1, sx2, y2])
                elif last_box is not None:
                    out_boxes.append([int(last_box[2]), int(last_box[1]),
                                      int(last_box[2]), int(last_box[3])])
                else:
                    fx1, fy1, _fx2, fy2 = fallback_box
                    out_boxes.append([fx1, fy1, fx1, fy2])
                out_chars.append(ch)
                continue
            if src_idx >= len(char_boxes):
                return None
            box = [int(v) for v in char_boxes[src_idx]]
            out_chars.append(ch)
            out_boxes.append(box)
            last_box = box
            src_idx += 1
        if src_idx == len(char_boxes) and len(out_chars) == len(text):
            return out_chars, out_boxes

    split = _split_box_by_units(text, fallback_box)
    if len(split) == len(text):
        return list(text), split
    return None


def _tighten_bbox_from_char_boxes(
    item: dict,
    image_size: tuple[int, int] | None,
) -> None:
    boxes = item.get("char_boxes")
    if not boxes:
        return
    valid = [b for b in boxes if len(b) == 4 and int(b[2]) > int(b[0])
             and int(b[3]) > int(b[1])]
    if not valid:
        return
    ux1 = min(int(b[0]) for b in valid)
    uy1 = min(int(b[1]) for b in valid)
    ux2 = max(int(b[2]) for b in valid)
    uy2 = max(int(b[3]) for b in valid)
    median_h = sorted(int(b[3]) - int(b[1]) for b in valid)[len(valid) // 2]
    pad_x = max(1, int(round(median_h * 0.08)))
    pad_y = max(1, int(round(median_h * 0.10)))
    width, height = image_size or (0, 0)
    item["baidu_bbox_original"] = [item["x1"], item["y1"], item["x2"], item["y2"]]
    item["x1"] = max(0, ux1 - pad_x)
    item["y1"] = max(0, uy1 - pad_y)
    item["x2"] = ux2 + pad_x if width <= 0 else min(width, ux2 + pad_x)
    item["y2"] = uy2 + pad_y if height <= 0 else min(height, uy2 + pad_y)


def _looks_like_baidu_watermark(item: dict, image_size: tuple[int, int] | None) -> bool:
    text = re.sub(r"\s+", "", str(item.get("text", "")))
    if not _WATERMARK_RE.match(text):
        return False
    # Keep normal page numbers and short numeric labels.
    if text.isdigit() and len(text) <= 4:
        return False
    if image_size is None:
        return True
    width, height = image_size
    x1, y1, x2, y2 = (int(item[k]) for k in ("x1", "y1", "x2", "y2"))
    near_edge = (
        x1 <= width * 0.08 or x2 >= width * 0.92
        or y1 <= height * 0.04 or y2 >= height * 0.94
    )
    return bool(near_edge or "86eb" in text.lower())


def _merge_items_on_same_line(left: dict, right: dict, sep: str) -> dict:
    out = dict(left)
    left_text = str(left.get("text", ""))
    right_text = str(right.get("text", ""))
    out["text"] = left_text + sep + right_text
    out["x1"] = min(int(left["x1"]), int(right["x1"]))
    out["y1"] = min(int(left["y1"]), int(right["y1"]))
    out["x2"] = max(int(left["x2"]), int(right["x2"]))
    out["y2"] = max(int(left["y2"]), int(right["y2"]))
    out["confidence"] = min(float(left.get("confidence", 1.0)),
                            float(right.get("confidence", 1.0)))
    out["ocr_backend"] = "baidu"
    out["baidu_merged"] = True

    chars: list[str] = []
    boxes: list[list[int]] = []
    for side, text in ((left, left_text), (right, right_text)):
        side_chars = side.get("chars")
        side_boxes = side.get("char_boxes")
        if side_chars is None or side_boxes is None or len(side_chars) != len(text):
            side_chars = list(text)
            side_boxes = _split_box_by_units(
                text, [side["x1"], side["y1"], side["x2"], side["y2"]])
        if side is right and sep:
            sx1 = int(left["x2"])
            sx2 = int(right["x1"])
            sy1 = min(int(left["y1"]), int(right["y1"]))
            sy2 = max(int(left["y2"]), int(right["y2"]))
            if sx2 <= sx1:
                sx1 = sx2 = int(round((int(left["x2"]) + int(right["x1"])) / 2))
            chars.extend(list(sep))
            boxes.extend([[sx1, sy1, sx2, sy2] for _ in sep])
        chars.extend(list(side_chars))
        boxes.extend([list(b) for b in side_boxes])
    if len(chars) == len(out["text"]) and len(boxes) == len(out["text"]):
        out["chars"] = chars
        out["char_boxes"] = boxes
        out["words"] = chars
        out["word_boxes"] = boxes
    return out


def _merge_footer_source_lines(
    items: list[dict],
    image_size: tuple[int, int] | None,
) -> list[dict]:
    if image_size is None:
        return items
    width, height = image_size
    bottom = [it for it in items if int(it["y1"]) >= height * 0.78]
    if not bottom:
        return items
    consumed: set[int] = set()
    merged_by_id: dict[int, dict] = {}
    indexed = list(enumerate(items))
    for idx, item in indexed:
        if idx in consumed or item not in bottom:
            continue
        text = str(item.get("text", ""))
        if text.strip() == "数据来源" or "http" not in text.lower() and "依据" not in text:
            continue
        line = item
        consumed.add(idx)
        changed = True
        while changed:
            changed = False
            cx = int(line["x2"])
            cy = (int(line["y1"]) + int(line["y2"])) / 2
            line_h = max(1, int(line["y2"]) - int(line["y1"]))
            candidates = []
            for j, other in indexed:
                if j in consumed or other not in bottom:
                    continue
                other_text = str(other.get("text", ""))
                if other_text.strip() == "数据来源":
                    continue
                oy = (int(other["y1"]) + int(other["y2"])) / 2
                gap = int(other["x1"]) - cx
                if abs(oy - cy) <= max(6, line_h * 0.45) and 0 <= gap <= width * 0.08:
                    candidates.append((gap, j, other))
            if not candidates:
                continue
            _gap, j, other = min(candidates, key=lambda p: p[0])
            sep = " " if (
                str(line.get("text", "")).strip()
                and str(other.get("text", "")).strip()
            ) else ""
            line = _merge_items_on_same_line(line, other, sep)
            consumed.add(j)
            changed = True
        merged_by_id[idx] = line

    if not merged_by_id:
        return items
    out: list[dict] = []
    for idx, item in enumerate(items):
        if idx in merged_by_id:
            out.append(merged_by_id[idx])
        elif idx not in consumed:
            out.append(item)
    return sorted(out, key=lambda it: (int(it["y1"]), int(it["x1"])))


def _postprocess_baidu_items(
    items: list[dict],
    image_size: tuple[int, int] | None,
) -> list[dict]:
    out: list[dict] = []
    for item in items:
        text = _normalize_symbols(str(item.get("text", "")))
        item["text"] = text
        chars = item.get("chars") or []
        boxes = item.get("char_boxes") or []
        aligned = _align_char_boxes(
            text, list(chars), [list(b) for b in boxes],
            [int(item["x1"]), int(item["y1"]), int(item["x2"]), int(item["y2"])],
        )
        if aligned is not None:
            item["chars"], item["char_boxes"] = aligned
            item["words"] = list(item["chars"])
            item["word_boxes"] = [list(b) for b in item["char_boxes"]]
            _tighten_bbox_from_char_boxes(item, image_size)
        if _looks_like_baidu_watermark(item, image_size):
            continue
        out.append(item)
    out = _merge_footer_source_lines(out, image_size)
    for item in out:
        if item.get("char_boxes"):
            _tighten_bbox_from_char_boxes(item, image_size)
    return out


@dataclass
class BaiduOcrConfig:
    api_key: str
    secret_key: str
    access_token: str
    endpoint: str = BAIDU_ACCURATE_ENDPOINT
    standard_endpoint: str = BAIDU_GENERAL_ENDPOINT
    provider_order: str = "high,standard"
    quota_fallback: bool = True
    token_url: str = BAIDU_TOKEN_URL
    language_type: str = "CHN_ENG"
    timeout_seconds: float = 30.0
    retries: int = 2

    def providers(self) -> list[BaiduOcrProvider]:
        aliases = {
            "high": BaiduOcrProvider("high", self.endpoint),
            "accurate": BaiduOcrProvider("high", self.endpoint),
            "hign": BaiduOcrProvider("high", self.endpoint),
            "standard": BaiduOcrProvider("standard", self.standard_endpoint),
            "general": BaiduOcrProvider("standard", self.standard_endpoint),
        }
        raw = (self.provider_order or "high").strip().lower()
        if raw in {"auto", "pool", "fallback"}:
            raw = "high,standard"
        out: list[BaiduOcrProvider] = []
        seen: set[tuple[str, str]] = set()
        for part in re.split(r"[,;\s]+", raw):
            if not part:
                continue
            provider = aliases.get(part)
            if provider is None:
                valid = ", ".join(sorted(aliases))
                raise BaiduOcrError(
                    f"Unknown Baidu OCR provider '{part}'. "
                    f"Use one of: {valid}."
                )
            key = (provider.name, provider.endpoint)
            if key not in seen:
                out.append(provider)
                seen.add(key)
        return out or [aliases["high"]]

    @classmethod
    def from_env(cls) -> "BaiduOcrConfig":
        return cls(
            api_key=_env(
                "DECKWEAVER_BAIDU_OCR_API_KEY",
                "BAIDU_OCR_API_KEY",
                "BAIDU_API_KEY",
            ),
            secret_key=_env(
                "DECKWEAVER_BAIDU_OCR_SECRET_KEY",
                "BAIDU_OCR_SECRET_KEY",
                "BAIDU_SECRET_KEY",
            ),
            access_token=_env(
                "DECKWEAVER_BAIDU_OCR_ACCESS_TOKEN",
                "BAIDU_OCR_ACCESS_TOKEN",
            ),
            endpoint=_env(
                "DECKWEAVER_BAIDU_OCR_ENDPOINT",
                "BAIDU_OCR_ENDPOINT",
                default=BAIDU_ACCURATE_ENDPOINT,
            ),
            standard_endpoint=_env(
                "DECKWEAVER_BAIDU_OCR_STANDARD_ENDPOINT",
                "BAIDU_OCR_STANDARD_ENDPOINT",
                default=BAIDU_GENERAL_ENDPOINT,
            ),
            provider_order=_env(
                "DECKWEAVER_BAIDU_OCR_PROVIDER_ORDER",
                "BAIDU_OCR_PROVIDER_ORDER",
                default="high,standard",
            ),
            quota_fallback=_env_bool(
                "DECKWEAVER_BAIDU_OCR_QUOTA_FALLBACK",
                "BAIDU_OCR_QUOTA_FALLBACK",
                default=True,
            ),
            token_url=_env(
                "DECKWEAVER_BAIDU_OCR_TOKEN_URL",
                "BAIDU_OCR_TOKEN_URL",
                default=BAIDU_TOKEN_URL,
            ),
            language_type=_env(
                "DECKWEAVER_BAIDU_OCR_LANGUAGE_TYPE",
                "BAIDU_OCR_LANGUAGE_TYPE",
                default="CHN_ENG",
            ),
            timeout_seconds=_env_float(
                "DECKWEAVER_BAIDU_OCR_TIMEOUT_SECONDS",
                "BAIDU_OCR_TIMEOUT_SECONDS",
                default=30.0,
            ),
            retries=_env_int(
                "DECKWEAVER_BAIDU_OCR_RETRIES",
                "BAIDU_OCR_RETRIES",
                default=2,
            ),
        )


class BaiduOcrClient:
    def __init__(self, config: BaiduOcrConfig):
        self.config = config
        self._token = config.access_token
        self._token_expiry = 0.0
        self.last_provider = ""
        self._disabled_providers: set[str] = set()

    def _post_json(
        self,
        url: str,
        *,
        data: bytes,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                req, timeout=self.config.timeout_seconds
            ) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            raise BaiduOcrError(f"Baidu OCR HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise BaiduOcrError(f"Baidu OCR network error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise BaiduOcrError("Baidu OCR request timed out") from exc
        try:
            decoded = payload.decode("utf-8")
            return json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BaiduOcrError("Baidu OCR returned invalid JSON") from exc

    def access_token(self) -> str:
        if self._token and time.time() < self._token_expiry:
            return self._token
        if self.config.access_token and self._token:
            return self._token
        if not self.config.api_key or not self.config.secret_key:
            raise BaiduOcrError(
                "Baidu OCR credentials are missing. Set "
                "DECKWEAVER_BAIDU_OCR_API_KEY and "
                "DECKWEAVER_BAIDU_OCR_SECRET_KEY."
            )
        query = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": self.config.api_key,
            "client_secret": self.config.secret_key,
        })
        payload = self._post_json(
            f"{self.config.token_url}?{query}",
            data=b"",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        token = payload.get("access_token")
        if not token:
            msg = payload.get("error_description") or payload.get("error") or payload
            raise BaiduOcrError(f"Baidu OAuth token request failed: {msg}")
        expires_in = int(payload.get("expires_in") or 0)
        self._token = str(token)
        # Refresh a minute early. Baidu currently returns a long-lived token,
        # but this also keeps short test tokens safe.
        self._token_expiry = time.time() + max(0, expires_in - 60)
        return self._token

    @staticmethod
    def _provider_disable_key(provider: BaiduOcrProvider) -> str:
        return f"{provider.name}:{provider.endpoint}"

    @staticmethod
    def _is_provider_fallback_error(exc: BaiduOcrApiError) -> bool:
        code = exc.error_code
        msg = exc.error_msg.lower()
        return (
            code in _PROVIDER_QUOTA_CODES
            or code in _PROVIDER_UNAVAILABLE_CODES
            or "quota" in msg
            or "daily request limit" in msg
            or "total request limit" in msg
            or "no permission" in msg
            or "service not support" in msg
        )

    @staticmethod
    def _is_retryable_error(exc: BaiduOcrApiError) -> bool:
        return exc.error_code in _RETRYABLE_CODES

    def _recognize_with_provider(
        self,
        provider: BaiduOcrProvider,
        *,
        token: str,
        body: bytes,
        min_conf: float,
        image_size: tuple[int, int] | None,
    ) -> list[dict]:
        url = f"{provider.endpoint}?access_token={urllib.parse.quote(token)}"
        last_error: BaiduOcrError | None = None
        for attempt in range(max(1, self.config.retries + 1)):
            try:
                payload = self._post_json(
                    url,
                    data=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                if "error_code" in payload:
                    raise BaiduOcrApiError(
                        payload.get("error_code"),
                        payload.get("error_msg"),
                        provider.name,
                    )
                items = extract_items(payload, min_conf, image_size=image_size)
                for item in items:
                    item["baidu_provider"] = provider.name
                self.last_provider = provider.name
                return items
            except BaiduOcrApiError as exc:
                last_error = exc
                if self._is_provider_fallback_error(exc):
                    break
                if not self._is_retryable_error(exc) or attempt >= self.config.retries:
                    break
                time.sleep(0.8 * (attempt + 1))
            except BaiduOcrError as exc:
                last_error = exc
                if attempt >= self.config.retries:
                    break
                time.sleep(0.6 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def recognize_image(self, image_path: Path, min_conf: float) -> list[dict]:
        token = self.access_token()
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        params = {
            "image": image_b64,
            "recognize_granularity": "small",
            "char_probability": "true",
            "probability": "true",
            "vertexes_location": "true",
            "paragraph": "false",
            "detect_direction": "false",
        }
        if self.config.language_type:
            params["language_type"] = self.config.language_type
        body = urllib.parse.urlencode(params).encode("utf-8")
        try:
            with Image.open(image_path) as im:
                image_size = im.size
        except Exception:
            image_size = None

        providers = self.config.providers()
        last_error: BaiduOcrError | None = None
        for idx, provider in enumerate(providers):
            disable_key = self._provider_disable_key(provider)
            if disable_key in self._disabled_providers:
                continue
            try:
                return self._recognize_with_provider(
                    provider,
                    token=token,
                    body=body,
                    min_conf=min_conf,
                    image_size=image_size,
                )
            except BaiduOcrApiError as exc:
                last_error = exc
                can_fallback = (
                    self.config.quota_fallback
                    and self._is_provider_fallback_error(exc)
                    and idx + 1 < len(providers)
                )
                if not can_fallback:
                    break
                self._disabled_providers.add(disable_key)
                print(
                    "  Baidu OCR provider "
                    f"{provider.name} unavailable ({exc.error_code}: "
                    f"{exc.error_msg}); falling back to {providers[idx + 1].name}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            except BaiduOcrError as exc:
                last_error = exc
                break
        assert last_error is not None
        raise last_error


def extract_items(
    payload: dict[str, Any],
    min_conf: float,
    image_size: tuple[int, int] | None = None,
) -> list[dict]:
    items: list[dict] = []
    for row in payload.get("words_result") or []:
        text = _normalize_symbols(str(row.get("words") or ""))
        if not text.strip():
            continue
        probability = row.get("probability") or {}
        conf = _normalize_conf(probability.get("average"), default=1.0)
        if conf < min_conf:
            continue
        bbox = (
            _bbox_from_points(row.get("vertexes_location"))
            or _bbox_from_location(row.get("location"))
        )
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue

        item = {
            "text": text,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": conf,
            "ocr_backend": "baidu",
        }
        chars = row.get("chars") or []
        char_texts: list[str] = []
        char_boxes: list[list[int]] = []
        char_confs: list[float] = []
        for ch in chars:
            c = str(ch.get("char") or "")
            if not c:
                continue
            cx1, cy1, cx2, cy2 = _bbox_from_location(ch.get("location"))
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            char_texts.append(c)
            char_boxes.append([cx1, cy1, cx2, cy2])
            char_confs.append(_normalize_conf(ch.get("char_prob"), default=conf))
        if (len(char_texts) == len(text)
                and _normalize_symbols("".join(char_texts)) == text):
            char_texts = list(text)
        aligned = _align_char_boxes(
            text,
            char_texts,
            char_boxes,
            [x1, y1, x2, y2],
        )
        if aligned is not None:
            item["chars"], item["char_boxes"] = aligned
            item["words"] = list(item["chars"])
            item["word_boxes"] = [list(b) for b in item["char_boxes"]]
            if len(char_confs) == len(item["chars"]):
                item["char_confidences"] = char_confs
        items.append(item)
    return _postprocess_baidu_items(items, image_size)


def run_ocr_batch(client: BaiduOcrClient, pairs: list[tuple[Path, Path]],
                  min_conf: float) -> list[int]:
    counts = []
    for img_path, out_path in pairs:
        out_path.unlink(missing_ok=True)
        if not img_path.exists():
            print(f"  SKIP missing: {img_path}", file=sys.stderr)
            counts.append(0)
            continue
        try:
            items = client.recognize_image(img_path, min_conf)
        except Exception as exc:
            print(f"  FAIL {img_path}: {exc}", file=sys.stderr)
            counts.append(0)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        counts.append(len(items))
        provider = client.last_provider or "unknown"
        print(
            f"  {img_path.name} -> {len(items)} items "
            f"(provider={provider})",
            flush=True,
        )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baidu OCR -> JSON.")
    parser.add_argument("inputs", nargs="*",
                        help="Single mode: one image path. Batch mode "
                             "(with --batch): pairs of image_path "
                             "out_json_path.")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode; write JSON to each out path.")
    parser.add_argument("--min-conf", type=float, default=0.3,
                        help="Drop detections below this confidence.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = BaiduOcrClient(BaiduOcrConfig.from_env())
    if args.batch:
        if len(args.inputs) % 2 != 0:
            sys.stderr.write("ERROR: --batch expects image/out_json pairs.\n")
            return 2
        pairs = [(Path(args.inputs[i]), Path(args.inputs[i + 1]))
                 for i in range(0, len(args.inputs), 2)]
        counts = run_ocr_batch(client, pairs, args.min_conf)
        return 0 if all(out.exists() for _, out in pairs) else 1

    if len(args.inputs) != 1:
        sys.stderr.write("ERROR: single mode expects exactly one image path.\n")
        return 2
    image_path = Path(args.inputs[0])
    if not image_path.exists():
        sys.stderr.write(f"ERROR: image not found: {image_path}\n")
        return 1
    try:
        items = client.recognize_image(image_path, args.min_conf)
    except BaiduOcrError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1
    print(json.dumps(items, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
