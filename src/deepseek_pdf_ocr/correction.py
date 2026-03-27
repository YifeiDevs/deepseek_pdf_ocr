# src/deepseek_pdf_ocr/correction.py
from __future__ import annotations
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from openai import OpenAI
from PIL import Image

OK_PLACEHOLDER = "<|ok|>"

_REF_DET_RE = re.compile(
    r"(<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>)", re.DOTALL
)
_IMAGE_REF_RE = re.compile(r"<\|ref\|>\s*image\s*<\|/ref\|>")
_REF_TYPE_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>")
_DET_COORDS_RE = re.compile(r"<\|det\|>(.*?)<\|/det\|>", re.DOTALL)

_SEARCH_REPLACE_RE = re.compile(
    r"<<<<\s*\n(.*?)\n\s*====\s*\n(.*?)\n\s*>>>>", re.DOTALL
)

_COORD_MAX = 999
_CROP_PADDING = 8

@dataclass
class Segment:
    index: int
    header: str       
    body: str         
    is_image: bool = False
    coords: list[list[int]] = field(default_factory=list)

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
    raw_response: str          
    raw_a: str = ""            
    raw_b: str = ""            
    summary: str = ""          # <-- 【新增】用于记录该页各段的对比表格
    n_segments: int = 0
    n_sent: int = 0
    n_ok: int = 0
    n_image_skipped: int = 0

    @property
    def n_corrected(self) -> int:
        return self.n_sent - self.n_ok

# --- 其他内部函数保持原有内容不变 ---

def _parse_coords(header: str) -> list[list[int]]:
    m = _DET_COORDS_RE.search(header)
    if not m:
        return []
    try:
        coords = eval(m.group(1).strip())
        if not coords:
            return []
        if isinstance(coords[0], int):
            return [coords]
        return [c for c in coords if isinstance(c, list) and len(c) == 4]
    except Exception:
        return []

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
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _apply_search_replace(original_body: str, corr_content: str, seg_idx: int) -> str:
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

def _correct_one_segment(
    seg: Segment,
    crop_imgs: list[Image.Image],
    page_image: Image.Image,
    client: OpenAI,
    model: str,
    temperature: float,
) -> tuple[int, str]:
    prompt = (
        _SINGLE_SEG_PROMPT
        .replace("<|ref_type|>", seg.ref_type or "text")
        .replace("<|ocr_text|>", seg.body.strip())
    )

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


def run_gpt_correction(
    ocr_result: str,
    image_path: str | Path,
    extracted_text: str,          
    api_key: str,
    endpoint: str,
    model: str = "gpt-5.2",
    temperature: float = 1.0,
    max_workers: int = 8,
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

    page_image = Image.open(image_path).convert("RGB")
    client = OpenAI(base_url=endpoint, api_key=api_key)

    crops: dict[int, list[Image.Image]] = {
        seg.index: _crop_segment_images(page_image, seg.coords)
        for seg in non_image
    }

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
                corrections[seg.index] = OK_PLACEHOLDER
                raw_parts.append(f"[{seg.index}] (error: {exc})")

    corrected = _reassemble(ocr_result, segments, corrections)
    n_ok = sum(1 for v in corrections.values() if v.strip() == OK_PLACEHOLDER)

    # =========================================================================
    # 按 segment index 顺序生成 A/B diff 文件内容 以及 summary 的 Markdown 差异表格
    # =========================================================================
    seg_by_idx = {s.index: s for s in segments}
    all_indices = sorted(seg_by_idx.keys())
    
    a_parts: list[str] = []
    b_parts: list[str] = []
    summary_parts: list[str] = []   # <-- 【新增】收集不同的部分，带有原图裁剪 HTML

    for idx in all_indices:
        seg = seg_by_idx[idx]
        
        if seg.is_image:
            a_parts.append(f"[{idx}]\n(image, skipped)")
            b_parts.append(f"[{idx}]\n(image, skipped)")
            continue

        gpt_reply = corrections.get(idx, OK_PLACEHOLDER)
        is_ok = gpt_reply.strip() == OK_PLACEHOLDER
        original_text = seg.body.strip()

        if is_ok:
            corrected_text = original_text
        elif "<<<<" in gpt_reply and "====" in gpt_reply and ">>>>" in gpt_reply:
            corrected_text = _apply_search_replace(seg.body, gpt_reply, seg.index).strip()
        else:
            corrected_text = gpt_reply.strip()

        if is_ok:
            a_parts.append(f"[{idx}]\n<|ok|>")
            b_parts.append(f"[{idx}]\n<|ok|>")
        elif original_text == corrected_text:
            a_parts.append(f"[{idx}]\n<|same|>")
            b_parts.append(f"[{idx}]\n<|same|>")
        else:
            # 文本确实发生了实质性改变
            a_parts.append(f"[{idx}]\n{original_text}")
            b_parts.append(f"[{idx}]\n{corrected_text}")

            # ======= 【新增】构建不同内容的 HTML 图片追踪以及 Markdown 表格 =======
            img_tags = []
            img_filename = Path(image_path).name
            W, H = page_image.size

            if not seg.coords:
                # 缺失坐标，直接引入整图
                img_tags.append(
                    f'<img src="../images_pages/{img_filename}" style="width: 100%; max-width: {W}px; height: auto;">'
                )
            else:
                for c in seg.coords:
                    x1 = max(0, int(c[0] / _COORD_MAX * W) - _CROP_PADDING)
                    y1 = max(0, int(c[1] / _COORD_MAX * H) - _CROP_PADDING)
                    x2 = min(W, int(c[2] / _COORD_MAX * W) + _CROP_PADDING)
                    y2 = min(H, int(c[3] / _COORD_MAX * H) + _CROP_PADDING)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    w_px = x2 - x1
                    h_px = y2 - y1
                    
                    # 【核心修复】：由于 object-fit: none 拒绝缩放，过宽的图在 PDF 会被截断。
                    # 设定一个安全的 A4 PDF 阅读宽度 (如 700px)。如果裁剪视口超过这个宽度，
                    # 我们利用 zoom 按比例整体缩放这个 HTML 元素，完美适应页面！
                    MAX_WIDTH = 700
                    scale = min(1.0, MAX_WIDTH / w_px) if w_px > 0 else 1.0
                    zoom_style = f" zoom: {scale:.3f};" if scale < 1.0 else ""
                    
                    # 注意必须加上 max-width: none; 防止被 Markdown 默认的 img { max-width: 100%; } 提前裁切
                    img_tags.append(
                        f'<img src="../images_pages/{img_filename}" style="width: {w_px}px; height: {h_px}px; object-fit: none; object-position: -{x1}px -{y1}px; max-width: none;{zoom_style}">'
                    )
            
            # 安全转义字符供在 Markdown Table 内多行使用
            def esc(txt: str) -> str:
                txt = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                return txt.replace("|", "&#124;").replace("\n", "<br>")
            
            # 使用 Markdown 生成对比表格
            summary_part = "\n".join(img_tags) + f"\n\n| Original | Modified |\n| --- | --- |\n| {esc(original_text)} | {esc(corrected_text)} |"
            summary_parts.append(summary_part)

    sep = "\n---\n"
    raw_a = sep.join(a_parts)
    raw_b = sep.join(b_parts)
    raw_response = sep.join(sorted(raw_parts))
    summary_md = "\n\n---\n\n".join(summary_parts) if summary_parts else ""

    return CorrectionResult(
        corrected=corrected,
        raw_response=raw_response,
        raw_a=raw_a,
        raw_b=raw_b,
        summary=summary_md,          # <-- 【新增】
        n_segments=len(segments),
        n_sent=len(non_image),
        n_ok=n_ok,
        n_image_skipped=n_image,
    )