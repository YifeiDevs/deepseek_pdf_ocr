"""GPT 校正：利用多模态 LLM 对 OCR 结果进行 **分段** 纠错。

v2 优化
-------
- 将 OCR 输出按 ``<|ref|>/<|det|>`` 标签切分为编号段落
- GPT 对已正确的段落只需返回 ``<|ok|>``，大幅节省 output token
- ``<|ref|>image<|/ref|>`` 段自动跳过，不发送给 GPT（无文字可校正）
- 只发段落类型标签（不暴露坐标），杜绝 GPT 误改坐标的风险

v3 优化 (Search & Replace)
-------
- 引入 `<<<< ==== >>>>` 局部替换语法。
- GPT 只需输出出错的那一两句话的修改，无需重写整个冗长的段落，速度翻倍。
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

# 匹配局部替换块 (Aider 风格)
# 允许界定符前后存在空白字符
_SEARCH_REPLACE_RE = re.compile(
    r"<<<<\s*\n(.*?)\n\s*====\s*\n(.*?)\n\s*>>>>", re.DOTALL
)


# ╔══════════════════════════════════════════════════════════════╗
# ║  Data structures                                             ║
# ╚══════════════════════════════════════════════════════════════╝


@dataclass
class Segment:
    index: int           
    header: str          
    body: str            
    is_image: bool = False   

    @property
    def ref_type(self) -> str:
        if not self.header:
            return ""
        m = _REF_TYPE_RE.search(self.header)
        return m.group(1).strip() if m else ""

    @property
    def full(self) -> str:
        return self.header + self.body


@dataclass
class CorrectionResult:
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
    parts = _REF_DET_RE.split(ocr_text)
    segments: list[Segment] = []
    idx = 1
    i = 0

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
    lines: list[str] = []
    gpt_to_abs: dict[int, int] = {}
    gpt_idx = 1

    for seg in segments:
        if seg.is_image:
            continue

        gpt_to_abs[gpt_idx] = seg.index

        type_label = seg.ref_type
        tag = f"[{gpt_idx} {type_label}]" if type_label else f"[{gpt_idx}]"
        lines.append(tag)
        lines.append(seg.body.strip())
        lines.append("")

        gpt_idx += 1

    return "\n".join(lines), gpt_to_abs


def _parse_gpt_response(response: str) -> dict[int, str]:
    markers = list(_SEG_NUM_RE.finditer(response))
    corrections: dict[int, str] = {}

    for i, m in enumerate(markers):
        seg_idx = int(m.group(1))
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(response)
        content = response[start:end].strip()
        corrections[seg_idx] = content

    return corrections


def _apply_search_replace(original_body: str, corr_content: str, seg_idx: int) -> str:
    """在段落原文中应用 `<<<< ==== >>>>` 替换块。"""
    blocks = _SEARCH_REPLACE_RE.findall(corr_content)
    new_body = original_body
    
    for search_text, replace_text in blocks:
        # 宽容处理：防止开头结尾有多余换行导致匹配失败
        search_target = search_text
        if search_target not in new_body:
            # 尝试 strip 后匹配
            if search_target.strip() in new_body:
                search_target = search_target.strip()
                replace_text = replace_text.strip()
            else:
                print(f"Warning: [GPT 校正] 在段落 [{seg_idx}] 中无法定位替换块:\n{search_target[:50]}...")
                continue
        
        # 只替换第一次出现（保证安全）
        new_body = new_body.replace(search_target, replace_text, 1)

    return new_body


def _reassemble(
    ocr_text: str,
    segments: list[Segment],
    corrections: dict[int, str],
) -> str:
    """将 GPT 校正结果合并回原始文本。"""
    result = ocr_text

    for seg in segments:
        if seg.is_image:
            continue

        corr = corrections.get(seg.index)
        if corr is None or corr.strip() == OK_PLACEHOLDER:
            continue

        old_full = seg.full

        leading_ws = ""
        for ch in seg.body:
            if ch in ("\n", "\r"):
                leading_ws += ch
            else:
                break

        body_rstripped = seg.body.rstrip()
        trailing_ws = seg.body[len(body_rstripped):]

        # 判断 GPT 是用了替换块，还是重写了整段
        if "<<<<" in corr and "====" in corr and ">>>>" in corr:
            # 局部替换模式
            modified_body = _apply_search_replace(seg.body, corr, seg.index)
            # 替换后维持原有前导和尾部空白
            new_full = seg.header + leading_ws + modified_body.strip() + trailing_ws
        else:
            # 全文重写模式 (Fallback)
            new_full = seg.header + leading_ws + corr.strip() + trailing_ws
            
        result = result.replace(old_full, new_full, 1)

    return result


# ╔══════════════════════════════════════════════════════════════╗
# ║  Prompt template                                             ║
# ╚══════════════════════════════════════════════════════════════╝

_CORRECTION_PROMPT_TEMPLATE = r"""# Role
你是一位精通学术文档 OCR 的**审校员**。你的职责是确保 OCR 结果在**内容上和科学上**的准确性，同时**尊重并保留原文的排版风格**。
# Task
下面是 OCR 引擎识别出的段落，请逐段检查并修正。

## 修正原则与禁区
### 你需要修正的：
1.  **明显拼写错误**：例如 OCR 识别混淆（`rn` ↔ `m`, `l` ↔ `1`, `O` ↔ `0`, `v` ↔ `y` ↔ `w`）。
2.  **严重损坏的 LaTeX**：例如括号不匹配、命令拼写错误。
3.  **明显的标点/语法错误**：影响句子理解的错误。

### 【重要】你绝对不能做的（保持风格，避免过度修正）：
1.  **禁止 LaTeX 化**：对于单位和化学式，如果原文可读，**绝对不要**画蛇添足地添加 `\mathrm` 或复杂的 LaTeX 命令。
    -   **示例**：`CaCO_{{3}}` 是正确的，**无需**改为 `\mathrm{{CaCO}}_{{3}}`。
2.  **禁止 Unicode 降级**：根据“以图像为准”的原则，如果图像中是 `m²`, `°C`, `μm`，那么它们就是**绝对正确**的。**严禁**因为参考文本是 `m2`, `C`, `um` 就进行降级修正。
3.  **忽略微小空格**：如果空格差异不影响数学或化学公式的含义，则**无需修正**。专注于实质性错误。

# Rules
1. **【最高优先级】** 若某段内容**完全正确**（根据上述原则），你**必须且只能**输出：`<|ok|>`。
2. **【局部修改】** 若只有少许错误，请使用**搜索替换块**。这是最快的方式。
   - **上块 (<<<<):** 必须从原文中**一字不差地复制**出包含错误的**最短片段**。
     - **要点1 (精确):** 包含所有原始空格和换行，否则程序会自动替换失败。
     - **要点2 (简短):** 只需保留错误处前后几个词作为上下文，严禁抄写长句。
     - **要点3 (禁止):** 绝对禁止使用 `...` 省略号。
   - **下块 (====):** 填写修正后的文本。
3. **【全文重写】** 只有当段落乱码严重、结构完全崩坏时，才直接输出整段修正文本。
4.  保留 HTML 表格结构不变。
5.  严格按照 `[N]` 编号逐段输出。

# OCR Segments

{numbered_segments}

# Reference (PDF直接提取的文字，仅作拼写校对辅助)

{extracted_text}

# Output Format
严格按编号输出，每段一个 `[N]`，后接 `<|ok|>`，或者替换块，或者全段内容。示例：

[1] <|ok|>

[2]
<<<<
Thls is a wrang sentence.
====
This is a wrong sentence.
>>>>

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
    if not ocr_result.strip():
        return CorrectionResult(corrected=ocr_result, raw_response="")

    segments = _parse_segments(ocr_result)
    if not segments:
        return CorrectionResult(corrected=ocr_result, raw_response="")

    n_image = sum(1 for s in segments if s.is_image)
    non_image = [s for s in segments if not s.is_image]

    if not non_image:
        return CorrectionResult(
            corrected=ocr_result,
            raw_response="(all segments are image references, nothing to correct)",
            n_segments=len(segments),
            n_sent=0,
            n_ok=0,
            n_image_skipped=n_image,
        )

    numbered_text, gpt_to_abs = _format_segments_for_prompt(segments)
    prompt = _CORRECTION_PROMPT_TEMPLATE.format(
        numbered_segments=numbered_text,
        extracted_text=extracted_text or "(无)",
    )

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

    gpt_corrections = _parse_gpt_response(raw_response)
    corrections: dict[int, str] = {}
    for gpt_idx, content in gpt_corrections.items():
        abs_idx = gpt_to_abs.get(gpt_idx)
        if abs_idx is not None:
            corrections[abs_idx] = content

    corrected = _reassemble(ocr_result, segments, corrections)

    n_ok = sum(1 for v in corrections.values() if v.strip() == OK_PLACEHOLDER)

    return CorrectionResult(
        corrected=corrected,
        raw_response=raw_response,
        n_segments=len(segments),
        n_sent=len(non_image),
        n_ok=n_ok,
        n_image_skipped=n_image,
    )