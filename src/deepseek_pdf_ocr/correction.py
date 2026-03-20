"""GPT 校正：利用多模态 LLM 对 OCR 结果进行 **分段** 纠错。

v2 优化
-------
- 将 OCR 输出按 ``<|ref|>/<|det|>`` 标签切分为编号段落
- GPT 对已正确的段落只需返回 ``<|ok|>``，大幅节省 output token
- ``<|ref|>image<|/ref|>`` 段自动跳过，不发送给 GPT（无文字可校正）
- 只发段落类型标签（不暴露坐标），杜绝 GPT 误改坐标的风险
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

# ╔══════════════════════════════════════════════════════════════╗
# ║  Constants                                                   ║
# ╚══════════════════════════════════════════════════════════════╝

OK_PLACEHOLDER = "<|ok|>"
"""GPT 返回此占位符表示"该段无需修改"。"""

_REF_DET_RE = re.compile(
    r"(<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>)", re.DOTALL
)

_IMAGE_REF_RE = re.compile(r"<\|ref\|>\s*image\s*<\|/ref\|>")
"""匹配 ``<|ref|>image<|/ref|>`` — 这类段落无文字内容，跳过校正。"""

_REF_TYPE_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>")
"""提取 ``<|ref|>`` 内的类型标签（如 text / equation / table）。"""

# 宽容解析：兼容 GPT 回复 "[1]" 和 "[1 figure_title]" 两种格式
_SEG_NUM_RE = re.compile(r"^\[(\d+)[^\]]*\]\s*", re.MULTILINE)


# ╔══════════════════════════════════════════════════════════════╗
# ║  Data structures                                             ║
# ╚══════════════════════════════════════════════════════════════╝


@dataclass
class Segment:
    """OCR 输出中的一个语义段落。"""

    index: int           # 在原始文本中的绝对序号（用于重组）
    header: str          # <|ref|>…<|/ref|><|det|>…<|/det|>  (前导文本为空串)
    body: str            # header 之后、下一个 header 之前的全部原始文本
    is_image: bool = False   # True → <|ref|>image<|/ref|> 段，跳过 GPT 校正

    @property
    def ref_type(self) -> str:
        """提取类型标签，如 ``text``, ``equation``, ``table``。"""
        if not self.header:
            return ""
        m = _REF_TYPE_RE.search(self.header)
        return m.group(1).strip() if m else ""

    @property
    def full(self) -> str:
        return self.header + self.body


@dataclass
class CorrectionResult:
    """GPT 校正的完整结果。

    Attributes
    ----------
    corrected : str
        完全还原的校正后文本（可直接写入 ``gpt5.2/``，与旧版等价）。
    raw_response : str
        GPT 原始回复（含 ``<|ok|>`` 占位符，写入 ``gpt5.2-raw/``）。
    n_segments : int
        总段落数（含 image 段）。
    n_sent : int
        实际发送给 GPT 的段落数（不含 image 段）。
    n_ok : int
        GPT 认为无需修改的段落数。
    n_image_skipped : int
        自动跳过的 image 段数量。
    """

    corrected: str
    raw_response: str
    n_segments: int = 0
    n_sent: int = 0
    n_ok: int = 0
    n_image_skipped: int = 0

    @property
    def n_corrected(self) -> int:
        return self.n_sent - self.n_ok


# ╔══════════════════════════════════════════════════════════════╗
# ║  Internal helpers                                            ║
# ╚══════════════════════════════════════════════════════════════╝


def _encode_image(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_segments(ocr_text: str) -> list[Segment]:
    """将 OCR 文本按 ``<|ref|>/<|det|>`` 标签切分为段落列表。

    ``re.split`` with a capturing group returns::

        [preamble, header₁, body₁, header₂, body₂, …]

    前导文本（preamble）如果非空也会产生一个 ``header=""`` 的段落。
    """
    parts = _REF_DET_RE.split(ocr_text)
    segments: list[Segment] = []
    idx = 1
    i = 0

    # 处理可能存在的前导文本
    if parts and not _REF_DET_RE.match(parts[0]):
        if parts[0].strip():
            segments.append(Segment(index=idx, header="", body=parts[0]))
            idx += 1
        i = 1

    while i < len(parts):
        header = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        is_image = bool(_IMAGE_REF_RE.search(header))
        segments.append(
            Segment(index=idx, header=header, body=body, is_image=is_image)
        )
        idx += 1
        i += 2

    return segments


def _format_segments_for_prompt(
    segments: list[Segment],
) -> tuple[str, dict[int, int]]:
    """将 **非 image** 段落格式化为 ``[N type]`` 编号文本，供 GPT 审阅。

    只发送类型标签（如 ``text`` / ``equation``），**不暴露坐标**，
    杜绝 GPT 误改坐标的风险。

    Returns
    -------
    formatted : str
        带 ``[N type]`` 编号的段落文本。
    gpt_to_abs : dict[int, int]
        ``{GPT 编号 → 绝对 segment.index}`` 映射表，
        用于将 GPT 回复映射回原始段落。
    """
    lines: list[str] = []
    gpt_to_abs: dict[int, int] = {}
    gpt_idx = 1

    for seg in segments:
        if seg.is_image:
            continue  # 跳过 image 段，不发送给 GPT

        gpt_to_abs[gpt_idx] = seg.index

        # [N type] 格式：紧凑且信息完整
        type_label = seg.ref_type
        tag = f"[{gpt_idx} {type_label}]" if type_label else f"[{gpt_idx}]"
        lines.append(tag)
        lines.append(seg.body.strip())
        lines.append("")  # 空行分隔

        gpt_idx += 1

    return "\n".join(lines), gpt_to_abs


def _parse_gpt_response(response: str) -> dict[int, str]:
    """解析 GPT 回复中的 ``[N]`` 或 ``[N type]`` 块。

    Returns
    -------
    dict[int, str]
        ``{GPT 编号: 校正内容 或 '<|ok|>'}``
    """
    markers = list(_SEG_NUM_RE.finditer(response))
    corrections: dict[int, str] = {}

    for i, m in enumerate(markers):
        seg_idx = int(m.group(1))
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(response)
        content = response[start:end].strip()
        corrections[seg_idx] = content

    return corrections


def _reassemble(
    ocr_text: str,
    segments: list[Segment],
    corrections: dict[int, str],
) -> str:
    """将 GPT 校正结果合并回原始文本。

    * image 段 → 原封不动保留
    * ``<|ok|>`` → 保留原文
    * 其他 → 替换 body，保留 header 与原始空白
    """
    result = ocr_text

    for seg in segments:
        if seg.is_image:
            continue  # image 段从未发给 GPT，跳过

        corr = corrections.get(seg.index)
        if corr is None or corr.strip() == OK_PLACEHOLDER:
            continue  # 原文不动

        old_full = seg.full

        # 保留 body 的前导换行
        leading_ws = ""
        for ch in seg.body:
            if ch in ("\n", "\r"):
                leading_ws += ch
            else:
                break

        # 保留 body 的尾部空白（段间间距）
        body_rstripped = seg.body.rstrip()
        trailing_ws = seg.body[len(body_rstripped):]

        new_full = seg.header + leading_ws + corr.strip() + trailing_ws
        result = result.replace(old_full, new_full, 1)

    return result


# ╔══════════════════════════════════════════════════════════════╗
# ║  Prompt template                                             ║
# ╚══════════════════════════════════════════════════════════════╝

_CORRECTION_PROMPT_TEMPLATE = r"""# Role
你是一位精通学术文档排版与 OCR 后处理的专家。

# Task
下面是 OCR 引擎从一张文档图片中识别出的结果，已按段落编号，格式为 `[N type]`。
- `N` 是段落编号
- `type` 是段落类型（如 text / equation / table / figure_title 等）
- 纯图片区域（`image` 类型）已被过滤，不在下方列表中。

请逐段检查并校正。

# Rules
1. **【最高优先级】** 若输入内容**完全正确且无需修改**，你**必须且只能**输出：`<|ok|>`；**禁止**复述/重写任何原文（用于节省 Token）。
2. 如果某段需要修正，请输出修正后的 **完整段落内容**（仅内容，不含 `[N type]` 标签头）。
3. 修正范围包括：
   - 拼写错误（OCR 常见混淆：l ↔ 1, rn ↔ m, O ↔ 0 等）
   - 标点符号
   - LaTeX 公式语法（上下标 `^` `_`、括号配对、命令拼写等）
4. 保留 HTML 表格结构（`<table>`, `<tr>` 等）不变。
5. **严格按照 `[N]` 编号逐段输出**，不要遗漏或增添编号。

# OCR Segments (image 段已过滤)

{numbered_segments}

# Reference (PDF 直接提取的文字，仅供参考)

{extracted_text}

# Output Format
严格按编号输出，每段一个 `[N]`，后跟 `<|ok|>` 或校正后的完整段落内容。示例：

[1] <|ok|>
[2] This is the corrected paragraph content...
[3] <|ok|>"""


# ╔══════════════════════════════════════════════════════════════╗
# ║  Public API                                                  ║
# ╚══════════════════════════════════════════════════════════════╝


def run_gpt_correction(
    ocr_result: str,
    image_path: str | Path,
    extracted_text: str,
    api_key: str,
    endpoint: str,
    model: str = "gpt-5.2",
    temperature: float = 1.0,
) -> CorrectionResult:
    """使用多模态 LLM 对 OCR 结果进行分段校正。

    ``<|ref|>image<|/ref|>`` 段自动跳过，不发送给 GPT。
    只发送类型标签，不暴露坐标，杜绝误改风险。

    Parameters
    ----------
    ocr_result : str
        DeepSeek OCR 原始输出。
    image_path : path-like
        对应页面图片路径。
    extracted_text : str
        PDF 内嵌文本（辅助校正参考）。
    api_key : str
        API Key。
    endpoint : str
        API endpoint / base_url。
    model : str
        模型名称。
    temperature : float
        采样温度。

    Returns
    -------
    CorrectionResult
        ``.corrected``    — 校正后完整文本（写入 ``gpt5.2/``）
        ``.raw_response`` — GPT 原始回复（写入 ``gpt5.2-raw/``）
    """
    # 0. 空输入保护
    if not ocr_result.strip():
        return CorrectionResult(
            corrected=ocr_result,
            raw_response="",
        )

    # 1. 分段
    segments = _parse_segments(ocr_result)

    if not segments:
        return CorrectionResult(
            corrected=ocr_result,
            raw_response="",
        )

    n_image = sum(1 for s in segments if s.is_image)
    non_image = [s for s in segments if not s.is_image]

    # 如果全部都是 image 段，无需调用 GPT
    if not non_image:
        return CorrectionResult(
            corrected=ocr_result,
            raw_response="(all segments are image references, nothing to correct)",
            n_segments=len(segments),
            n_sent=0,
            n_ok=0,
            n_image_skipped=n_image,
        )

    # 2. 构建 prompt（image 段已过滤，坐标不暴露）
    numbered_text, gpt_to_abs = _format_segments_for_prompt(segments)
    prompt = _CORRECTION_PROMPT_TEMPLATE.format(
        numbered_segments=numbered_text,
        extracted_text=extracted_text or "(无)",
    )

    # 3. 调用 GPT
    client = OpenAI(base_url=endpoint, api_key=api_key)
    image_base64 = _encode_image(image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
        temperature=temperature,
    )

    raw_response = response.choices[0].message.content

    # 4. 解析回复 — GPT 编号 → 绝对 segment index
    gpt_corrections = _parse_gpt_response(raw_response)
    corrections: dict[int, str] = {}
    for gpt_idx, content in gpt_corrections.items():
        abs_idx = gpt_to_abs.get(gpt_idx)
        if abs_idx is not None:
            corrections[abs_idx] = content

    # 5. 合并
    corrected = _reassemble(ocr_result, segments, corrections)

    # 6. 统计
    n_ok = sum(1 for v in corrections.values() if v.strip() == OK_PLACEHOLDER)

    return CorrectionResult(
        corrected=corrected,
        raw_response=raw_response,
        n_segments=len(segments),
        n_sent=len(non_image),
        n_ok=n_ok,
        n_image_skipped=n_image,
    )