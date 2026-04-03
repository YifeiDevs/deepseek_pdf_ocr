"""极简规则 + LLM 复核 的 OCR 修改过滤器。"""

from __future__ import annotations

import base64
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Sequence

from openai import OpenAI
from PIL import Image

from deepseek_pdf_ocr.prompt_loader import load_markdown_messages


# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────

PROMPT_MD_PATH = Path(__file__).parent / "prompt" / "judge_change_filter.md"


@lru_cache(maxsize=1)
def _load_prompt_template() -> list[dict]:
    return load_markdown_messages(PROMPT_MD_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# 轻量本地规则
# ──────────────────────────────────────────────────────────────────────────────

_QUOTE_TRANS = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "＂": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "`": "'",
        "´": "'",
    }
)

_DASH_TRANS = str.maketrans(
    {
        "‐": "-",   # hyphen
        "‑": "-",   # non-breaking hyphen
        "‒": "-",   # figure dash
        "–": "-",   # en dash
        "—": "-",   # em dash
        "―": "-",   # horizontal bar
        "−": "-",   # minus sign
        "－": "-",   # fullwidth hyphen-minus
    }
)


def _remove_ws(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _normalize_quotes_dashes(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_QUOTE_TRANS)
    text = text.translate(_DASH_TRANS)
    return text


def _strip_md_heading(text: str) -> str:
    # 仅移除行首 Markdown 标题标记，如 # / ## / ###
    return re.sub(r"(?m)^[ \t]*#{1,6}[ \t]+", "", text or "")


def _collapse_linebreak_hyphen(text: str) -> str:
    # therefore  <->  there- fore
    # mono-\nsulfate <-> monosulfate
    return re.sub(r"(?<=\w)-\s+(?=\w)", "", text or "")


def detect_trivial_change(original_text: str, candidate_text: str) -> str | None:
    """返回命中的极简规则名；若不是明显格式改动则返回 None。"""
    original = (original_text or "").strip()
    candidate = (candidate_text or "").strip()

    if not original or not candidate:
        return None

    if original == candidate:
        return "exact_same"

    # 1) 只改空格
    if _remove_ws(original) == _remove_ws(candidate):
        return "whitespace_only"

    # 2) 只改引号 / dash / 空格
    o1 = _normalize_quotes_dashes(original)
    c1 = _normalize_quotes_dashes(candidate)
    if _remove_ws(o1) == _remove_ws(c1):
        return "quotes_or_dashes_only"

    # 3) 只改 Markdown 标题 ## / ### / ...
    o2 = _strip_md_heading(o1)
    c2 = _strip_md_heading(c1)
    if _remove_ws(o2) == _remove_ws(c2):
        return "markdown_heading_only"

    # 4) 只把完整词拆回跨行断词
    o3 = _collapse_linebreak_hyphen(o2)
    c3 = _collapse_linebreak_hyphen(c2)
    if _remove_ws(o3) == _remove_ws(c3):
        return "linebreak_hyphen_only"

    return None


# ──────────────────────────────────────────────────────────────────────────────
# LLM 复核
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class FilterDecision:
    keep: bool
    source: str       # "rule" | "llm" | "fallback"
    reason: str = ""
    raw_response: str = ""


def _image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_image_parts(
    crop_imgs: Sequence[Image.Image] | None,
    page_image: Image.Image | None,
    max_images: int,
) -> list[dict]:
    imgs = list(crop_imgs or [])
    if not imgs and page_image is not None:
        imgs = [page_image]

    parts: list[dict] = []
    for i, img in enumerate(imgs[:max_images], start=1):
        parts.append({"type": "text", "text": f"[Image {i}]"})
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_image_to_base64(img)}"},
            }
        )
    return parts


def _materialize_messages(
    messages_template: list[dict],
    *,
    original_text: str,
    candidate_text: str,
    image_parts: list[dict],
) -> list[dict]:
    """将 prompt 模板中的占位符替换为实际内容。"""
    messages = deepcopy(messages_template)

    for msg in messages:
        new_content: list[dict] = []
        for item in msg.get("content", []):
            if item.get("type") != "text":
                new_content.append(item)
                continue

            text = item.get("text", "")

            # 先替换图片占位符
            if "<|ImagePlaceholder|>" in text:
                parts = text.split("<|ImagePlaceholder|>")
                for idx, part in enumerate(parts):
                    part = (
                        part.replace("<|OriginalText|>", original_text)
                        .replace("<|CandidateText|>", candidate_text)
                    )
                    if part.strip():
                        new_content.append({"type": "text", "text": part.strip()})
                    if idx < len(parts) - 1:
                        new_content.extend(image_parts)
                continue

            # 再替换普通文本占位符
            text = (
                text.replace("<|OriginalText|>", original_text)
                .replace("<|CandidateText|>", candidate_text)
            )
            if text.strip():
                new_content.append({"type": "text", "text": text.strip()})

        msg["content"] = new_content

    return messages


def _parse_llm_response(raw: str) -> tuple[bool | None, str]:
    text = (raw or "").strip()
    if not text:
        return None, "empty_response"

    # 去掉 code fence
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)

    # 优先解析 JSON
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            decision = str(data.get("decision", "")).strip().upper()
            reason = str(data.get("reason", "")).strip()
            if decision in {"KEEP", "DROP"}:
                return decision == "KEEP", (reason or "llm_json")
        except Exception:
            pass

    # 退化到关键词搜索
    m = re.search(r"\b(KEEP|DROP)\b", text.upper())
    if m:
        return m.group(1) == "KEEP", "llm_plain_text"

    return None, "unparseable_response"


def review_change(
    *,
    original_text: str,
    candidate_text: str,
    client: OpenAI | None = None,
    model: str | None = None,
    crop_imgs: Sequence[Image.Image] | None = None,
    page_image: Image.Image | None = None,
    temperature: float = 0.0,
    max_images: int = 4,
    fail_open: bool = False,
) -> FilterDecision:
    """判断候选修改是否应保留。"""
    original_text = (original_text or "").strip()
    candidate_text = (candidate_text or "").strip()

    # 先跑极简规则
    trivial_reason = detect_trivial_change(original_text, candidate_text)
    if trivial_reason is not None:
        return FilterDecision(
            keep=False,
            source="rule",
            reason=trivial_reason,
        )

    # 没有 judge 配置时，只做本地规则
    if client is None or not model:
        return FilterDecision(
            keep=True,
            source="fallback",
            reason="judge_disabled",
        )

    try:
        messages_template = _load_prompt_template()
        image_parts = _build_image_parts(
            crop_imgs=crop_imgs,
            page_image=page_image,
            max_images=max_images,
        )
        messages = _materialize_messages(
            messages_template,
            original_text=original_text,
            candidate_text=candidate_text,
            image_parts=image_parts,
        )

        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
        )

        raw = response.choices[0].message.content or ""
        keep, reason = _parse_llm_response(raw)

        if keep is None:
            return FilterDecision(
                keep=fail_open,
                source="fallback",
                reason=reason,
                raw_response=raw,
            )

        return FilterDecision(
            keep=keep,
            source="llm",
            reason=reason,
            raw_response=raw,
        )

    except Exception as exc:
        return FilterDecision(
            keep=fail_open,
            source="fallback",
            reason=f"judge_error: {exc}",
        )


def filter_candidate_text(
    *,
    original_text: str,
    candidate_text: str,
    client: OpenAI | None = None,
    model: str | None = None,
    crop_imgs: Sequence[Image.Image] | None = None,
    page_image: Image.Image | None = None,
    temperature: float = 0.0,
    max_images: int = 4,
    fail_open: bool = False,
) -> tuple[str, FilterDecision]:
    """返回过滤后的最终文本以及判定详情。"""
    decision = review_change(
        original_text=original_text,
        candidate_text=candidate_text,
        client=client,
        model=model,
        crop_imgs=crop_imgs,
        page_image=page_image,
        temperature=temperature,
        max_images=max_images,
        fail_open=fail_open,
    )

    final_text = candidate_text if decision.keep else original_text
    return final_text, decision


def should_keep_change(**kwargs) -> bool:
    """review_change 的布尔快捷封装。"""
    return review_change(**kwargs).keep


__all__ = [
    "FilterDecision",
    "detect_trivial_change",
    "review_change",
    "filter_candidate_text",
    "should_keep_change",
]