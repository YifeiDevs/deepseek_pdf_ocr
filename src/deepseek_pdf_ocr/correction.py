"""GPT 校正：利用多模态 LLM 对 OCR 结果进行 **逐段** 纠错。
v4 重构（按段裁图 + 并行）
-------
- 每个非 image 的 segment，根据 <|det|> 坐标从整页大图中裁出对应小图
- 多框 segment（如跨列文本：左下角一块 + 右上角一块）按坐标顺序各自裁剪，
  依次传给 GPT，而不是合并成大框（合并会把两列之间的空白一起框进去）
- 将「小图(s) + 该段 OCR 文本」发送给 GPT，聚焦视野，避免整页干扰
- 多个 segment 并发请求（ThreadPoolExecutor），吞吐量与 DeepSeek 批处理相当
- 去掉 extracted_text（Reference）：小段无法对齐，意义不大
- GPT 只需直接输出修正后的文本，或输出 <|ok|> 表示无需修改
- 保留 Search & Replace 语法以节省 output token（可选）
"""
from __future__ import annotations
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from openai import OpenAI
from PIL import Image

# ╔══════════════════════════════════════════════════════════════╗
# ║  Constants                                                   ║
# ╚══════════════════════════════════════════════════════════════╝
OK_PLACEHOLDER = "<|ok|>"
"""GPT 返回此占位符表示"该段无需修改"。" """

_REF_DET_RE = re.compile(
    r"(<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>)", re.DOTALL
)
_IMAGE_REF_RE = re.compile(r"<\|ref\|>\s*image\s*<\|/ref\|>")
_REF_TYPE_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>")
_DET_COORDS_RE = re.compile(r"<\|det\|>(.*?)<\|/det\|>", re.DOTALL)

# Search & Replace 块
_SEARCH_REPLACE_RE = re.compile(
    r"<<<<\s*\n(.*?)\n\s*====\s*\n(.*?)\n\s*>>>>", re.DOTALL
)

# 归一化坐标范围（DeepSeek OCR 使用 0-999）
_COORD_MAX = 999
# 裁图时的外边距（像素），避免贴边
_CROP_PADDING = 8

# ╔══════════════════════════════════════════════════════════════╗
# ║  Data structures                                             ║
# ╚══════════════════════════════════════════════════════════════╝
@dataclass
class Segment:
    index: int
    header: str       # <|ref|>...<|/ref|><|det|>...<|/det|>
    body: str         # 段落文字内容
    is_image: bool = False
    coords: list[list[int]] = field(default_factory=list)
    """所有检测框坐标列表，每项为 [x1, y1, x2, y2]（0-999 归一化）。"""

    @property
    def ref_type(self) -> str:
        m = _REF_TYPE_RE.search(self.header)
        return m.group(1).strip() if m else ""

    @property
    def full(self) -> str:
        return self.header + self.body


@dataclass
class CorrectionResult:
    corrected: str
    raw_response: str          # 兼容旧字段，保留
    raw_a: str = ""            # diff 左侧：修正前的原文
    raw_b: str = ""            # diff 右侧：修正后的文本
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
def _parse_coords(header: str) -> list[list[int]]:
    """从 header 中提取所有检测框坐标。"""
    m = _DET_COORDS_RE.search(header)
    if not m:
        return []
    try:
        coords = eval(m.group(1).strip())  # noqa: S307
        if not coords:
            return []
        # 支持单框 [[x1,y1,x2,y2]] 和多框 [[...],[...]]
        if isinstance(coords[0], int):
            return [coords]
        return [c for c in coords if isinstance(c, list) and len(c) == 4]
    except Exception:
        return []

def _parse_segments(ocr_text: str) -> list[Segment]:
    """将 OCR 文本按 ref/det 标签切分为 Segment 列表。"""
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
        coords = _parse_coords(header)
        segments.append(
            Segment(
                index=idx,
                header=header,
                body=body,
                is_image=is_image,
                coords=coords,
            )
        )
        idx += 1
        i += 2
    return segments

def _crop_segment_images(
    page_image: Image.Image,
    coords: list[list[int]],
    padding: int = _CROP_PADDING,
) -> list[Image.Image]:
    """根据归一化坐标从整页图像中裁剪出该段的所有检测框，按顺序返回。
    多框情况（如跨列文本：左下角一块 + 右上角一块）不做合并，
    而是按坐标列表顺序各自裁剪，由调用方按顺序传给 GPT。
    这样 GPT 能分别看清每一块的实际内容，避免框选两列之间的空白区域。
    """
    if not coords:
        return []
    W, H = page_image.size
    crops: list[Image.Image] = []
    for c in coords:
        x1 = max(0, int(c[0] / _COORD_MAX * W) - padding)
        y1 = max(0, int(c[1] / _COORD_MAX * H) - padding)
        x2 = min(W, int(c[2] / _COORD_MAX * W) + padding)
        y2 = min(H, int(c[3] / _COORD_MAX * H) + padding)
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(page_image.crop((x1, y1, x2, y2)))
    return crops

def _image_to_base64(img: Image.Image) -> str:
    """PIL Image → base64 PNG 字符串。"""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _apply_search_replace(original_body: str, corr_content: str, seg_idx: int) -> str:
    """在段落原文中应用 `<<<< ==== >>>>` 替换块。"""
    blocks = _SEARCH_REPLACE_RE.findall(corr_content)
    new_body = original_body
    for search_text, replace_text in blocks:
        search_target = search_text
        if search_target not in new_body:
            if search_target.strip() in new_body:
                search_target = search_target.strip()
                replace_text = replace_text.strip()
            else:
                print(
                    f"Warning: [GPT 校正] 在段落 [{seg_idx}] 中无法定位替换块:\n"
                    f"{search_target[:50]}..."
                )
                continue
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

        if "<<<<" in corr and "====" in corr and ">>>>" in corr:
            modified_body = _apply_search_replace(seg.body, corr, seg.index)
            new_full = seg.header + leading_ws + modified_body.strip() + trailing_ws
        else:
            new_full = seg.header + leading_ws + corr.strip() + trailing_ws

        result = result.replace(old_full, new_full, 1)

    return result

# ╔══════════════════════════════════════════════════════════════╗
# ║  Prompt template（单段）                                     ║
# ╚══════════════════════════════════════════════════════════════╝
_SINGLE_SEG_PROMPT = r"""# Role
你是一位精通学术文档 OCR 的**校对员**，任务是纠正 OCR 识别错误，而不是改写或美化文本。

# Task
下面提供了从 PDF 页面裁剪出的**区域图像**（若有多张，说明该段文字跨列分布，请按顺序阅读），
以及对应的 OCR 识别文本。**图像是最终权威**，请严格以图像内容为准进行校对。

## 校对原则

### 【必须修正】真正的 OCR 识别错误：
1. 字符混淆（`rn`↔`m`，`l`↔`1`，`O`↔`0`，`v`↔`y`↔`w` 等视觉相似字符）
2. LaTeX 命令拼写错误（如 `\frc` 应为 `\frac`）或严重的括号不匹配
3. 明显的单词拼写错误（非风格差异，是真实识别错误）

### 【严禁修改】以下情况即使你认为"更好"也绝对不能动：
1. **上标、下标、角标**：图中有 `^{15}`、`_n` 等，OCR 文本里有就保留，绝对不能删除或移动
2. **序号与编号**：图中的序号是什么就是什么，即使看起来"错误"也以图为准
3. **空格**：不得以"优化显示"为由增删空格，空格差异不是 OCR 错误
4. **LaTeX 风格**：`CO_2` 和 `\mathrm{CO}_2` 都是合法写法，不得互相转换；不得添加 `\mathrm`、`\text`、`\boldsymbol` 等装饰命令
5. **数学符号呈现方式**：`\( x \)` 和 `$x$` 等价，不得转换格式
6. **HTML 表格**：不得将 HTML 表格改写为 Markdown 表格，反之亦然
7. **标点风格**：连字符、破折号等的具体形式以图为准，不得"标准化"

**判断准则**：如果你在考虑"这样改是不是更规范/更好看"，那就不要改。只改"这里识别错了"的情况。

## OCR 文本（类型：<|ref_type|>）
<|ocr_text|>

## Output 格式
- 若文本**完全正确**（或仅有无关紧要的空格差异），仅输出：`<|ok|>`
- 若只有少许错误，使用**搜索替换块**，精确定位出错位置：
``
<<<<
原文中有误的最短片段（必须与原文完全一致，包括空格和换行）
====
修正后的文本
>>>>
``
- 若段落严重损坏（大量字符错误），直接输出整段修正文本。
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  Single-segment GPT call                                     ║
# ╚══════════════════════════════════════════════════════════════╝
def _correct_one_segment(
    seg: Segment,
    crop_imgs: list[Image.Image],
    page_image: Image.Image,
    client: OpenAI,
    model: str,
    temperature: float,
) -> tuple[int, str]:
    """对单个 segment 调用 GPT，返回 (seg.index, corrected_text)。
    多框 segment（如跨列文本）按坐标顺序发送多张裁剪图：
      - 第 1 张图 = 第 1 个检测框的裁剪区域
      - 第 2 张图 = 第 2 个检测框的裁剪区域（如右列续文）
      - ……
    GPT 依次看到各块后结合 OCR 文本进行校正。
    若无法裁图（坐标缺失）则回退到整页图。
    """
    prompt = (
        _SINGLE_SEG_PROMPT
        .replace("<|ref_type|>", seg.ref_type or "text")
        .replace("<|ocr_text|>", seg.body.strip())
    )

    # 构建 content：先放 prompt 文字，再按顺序放裁剪图（或整页图兜底）
    content: list[dict] = [{"type": "text", "text": prompt}]
    if crop_imgs:
        for img in crop_imgs:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{_image_to_base64(img)}"
                    },
                }
            )
    else:
        # 没有坐标信息，回退到整页图
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_image_to_base64(page_image)}"
                },
            }
        )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
    )

    return seg.index, response.choices[0].message.content

# ╔══════════════════════════════════════════════════════════════╗
# ║  Public API                                                  ║
# ╚══════════════════════════════════════════════════════════════╝
def run_gpt_correction(
    ocr_result: str,
    image_path: str | Path,
    extracted_text: str,          # 保留参数签名兼容性，不再使用
    api_key: str,
    endpoint: str,
    model: str = "gpt-5.2",
    temperature: float = 1.0,
    max_workers: int = 8,
) -> CorrectionResult:
    """对整页 OCR 结果按 segment 并发校正。"""

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

    # 加载整页图像（用于裁剪）
    page_image = Image.open(image_path).convert("RGB")
    client = OpenAI(base_url=endpoint, api_key=api_key)

    # 预裁剪所有 segment 的小图列表（多框时保留多张，顺序与坐标列表一致）
    crops: dict[int, list[Image.Image]] = {
        seg.index: _crop_segment_images(page_image, seg.coords)
        for seg in non_image
    }

    # 并发请求 GPT
    corrections: dict[int, str] = {}
    raw_parts: list[str] = []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(non_image))) as pool:
        futures = {
            pool.submit(
                _correct_one_segment,
                seg,
                crops[seg.index],
                page_image,
                client,
                model,
                temperature,
            ): seg
            for seg in non_image
        }

        for future in as_completed(futures):
            seg = futures[future]
            try:
                idx, content = future.result()
                corrections[idx] = content
                raw_parts.append(f"[{idx}] {content}")
            except Exception as exc:
                print(f"  Warning: segment [{seg.index}] GPT 校正失败: {exc}")
                # 失败时保留原文（等同于 <|ok|>）
                corrections[seg.index] = OK_PLACEHOLDER
                raw_parts.append(f"[{seg.index}] (error: {exc})")

    corrected = _reassemble(ocr_result, segments, corrections)
    n_ok = sum(1 for v in corrections.values() if v.strip() == OK_PLACEHOLDER)

    # =========================================================================
    # 按 segment index 顺序生成 A/B diff 文件内容
    # A 侧：修正前的原文
    # B 侧：修正后的文本
    # 策略：
    # - 若完全没有修改（GPT返回 <|ok|>，或替换内容完全没变），两边都写入 <|same|>
    # - 若发生了实际修改，A写更正前，B写更正后的【最终实际文本】（应用替换后）
    # =========================================================================
    seg_by_idx = {s.index: s for s in segments}
    all_indices = sorted(seg_by_idx.keys())
    a_parts: list[str] = []
    b_parts: list[str] = []

    for idx in all_indices:
        seg = seg_by_idx[idx]
        
        # 1. Image 占位，直接跳过
        if seg.is_image:
            a_parts.append(f"[{idx}]\n(image, skipped)")
            b_parts.append(f"[{idx}]\n(image, skipped)")
            continue

        gpt_reply = corrections.get(idx, OK_PLACEHOLDER)
        is_ok = gpt_reply.strip() == OK_PLACEHOLDER
        original_text = seg.body.strip()

        # 2. 计算修正后的最终文本
        if is_ok:
            corrected_text = original_text
        elif "<<<<" in gpt_reply and "====" in gpt_reply and ">>>>" in gpt_reply:
            corrected_text = _apply_search_replace(seg.body, gpt_reply, seg.index).strip()
        else:
            corrected_text = gpt_reply.strip()

        # 3. 比较差异并写入 A/B (区分 <|ok|> 和 <|same|>)
        if is_ok:
            # GPT 明确认为不需要修改，返回了 <|ok|>
            a_parts.append(f"[{idx}]\n<|ok|>")
            b_parts.append(f"[{idx}]\n<|ok|>")
        elif original_text == corrected_text:
            # GPT 输出了修正内容（或替换块），但应用后与原文一模一样
            a_parts.append(f"[{idx}]\n<|same|>")
            b_parts.append(f"[{idx}]\n<|same|>")
        else:
            # 文本确实发生了实质性改变
            a_parts.append(f"[{idx}]\n{original_text}")
            b_parts.append(f"[{idx}]\n{corrected_text}")

    sep = "\n---\n"
    raw_a = sep.join(a_parts)
    raw_b = sep.join(b_parts)
    # raw_response 保持旧格式兼容
    raw_response = sep.join(sorted(raw_parts))

    return CorrectionResult(
        corrected=corrected,
        raw_response=raw_response,
        raw_a=raw_a,
        raw_b=raw_b,
        n_segments=len(segments),
        n_sent=len(non_image),
        n_ok=n_ok,
        n_image_skipped=n_image,
    )