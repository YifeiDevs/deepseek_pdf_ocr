# src/deepseek_pdf_ocr/correction.py
from __future__ import annotations
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from openai import OpenAI
from PIL import Image

# 引入现有的 prompt loader
from deepseek_pdf_ocr.prompt_loader import load_markdown_messages

OK_PLACEHOLDER = "<|ok|>"

# 定位 few_shot_prompt.md 的绝对路径
PROMPT_MD_PATH = Path(__file__).parent / "prompt" / "few_shot_prompt.md"

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
    summary: str = ""          
    readable_summary: str = "" # <-- 【新增】用于记录新格式的易读版本
    n_segments: int = 0
    n_sent: int = 0
    n_ok: int = 0
    n_image_skipped: int = 0

    @property
    def n_corrected(self) -> int:
        return self.n_sent - self.n_ok


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
        if corr is None or OK_PLACEHOLDER in corr or "<|abort|>" in corr:
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

def _correct_one_segment(
    seg: Segment,
    crop_imgs: list[Image.Image],
    page_image: Image.Image,
    client: OpenAI,
    model: str,
    temperature: float,
    messages_template: list[dict], # <-- 【新增】传入已加载的 few-shot 模板
) -> tuple[int, str]:
    
    # 深度拷贝，防止多线程污染模板
    messages = deepcopy(messages_template)
    
    # 构建当前段落的图片数据字典
    img_dicts = []
    if crop_imgs:
        for img in crop_imgs:
            img_dicts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_image_to_base64(img)}"
                },
            })
    else:
        img_dicts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{_image_to_base64(page_image)}"
            },
        })

    # 定位并替换最后的 User 消息中的占位符
    last_user_msg = messages[-1]
    new_content = []
    
    for item in last_user_msg["content"]:
        if item["type"] == "text":
            text = item["text"]
            
            # 替换 ImagePlaceholder 为实际的图像 dict 列表
            if "<|ImagePlaceholder|>" in text:
                parts = text.split("<|ImagePlaceholder|>")
                if parts[0].strip():
                    new_content.append({"type": "text", "text": parts[0].strip()})
                
                new_content.extend(img_dicts)
                text = parts[1] # 继续处理剩下的文本
                
            # 替换 TextPlaceholder 为当前 OCR 原文
            if "<|TextPlaceholder|>" in text:
                text = text.replace("<|TextPlaceholder|>", seg.body.strip())
                
            if text.strip():
                new_content.append({"type": "text", "text": text.strip()})
        else:
            new_content.append(item)
            
    last_user_msg["content"] = new_content

    response = client.chat.completions.create(
        model=model,
        messages=messages,
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

    # 预加载 Few-Shot Markdown 模板（一次解析，线程共享深度拷贝）
    messages_template = load_markdown_messages(PROMPT_MD_PATH)

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
                messages_template, # 传入模板
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
    # 【修改点】同样使用 in 判断
    n_ok = sum(1 for v in corrections.values() if OK_PLACEHOLDER in v or "<|abort|>" in v)

    # =========================================================================
    # 生成 A/B diff 文件内容 以及 summary 的 Markdown 表格 和 易读排版版
    # =========================================================================
    seg_by_idx = {s.index: s for s in segments}
    all_indices = sorted(seg_by_idx.keys())
    
    a_parts: list[str] = []
    b_parts: list[str] = []
    summary_parts: list[str] = []   
    readable_parts: list[str] = []  # <-- 【新增】收集易读的 Markdown 格式

    for idx in all_indices:
        seg = seg_by_idx[idx]
        
        if seg.is_image:
            a_parts.append(f"[{idx}]\n(image, skipped)")
            b_parts.append(f"[{idx}]\n(image, skipped)")
            continue

        gpt_reply = corrections.get(idx, OK_PLACEHOLDER)
        # 【修改点】改为基于 in 判断
        is_ok = OK_PLACEHOLDER in gpt_reply or "<|abort|>" in gpt_reply
        original_text = seg.body.strip()

        if is_ok:
            corrected_text = original_text
        elif "<<<<" in gpt_reply and "====" in gpt_reply and ">>>>" in gpt_reply:
            corrected_text = _apply_search_replace(seg.body, gpt_reply, seg.index).strip()
        else:
            corrected_text = gpt_reply.strip()

        if is_ok:
            a_parts.append(f"[{idx}]\n<|ok|>")
            # 【优化】即使跳过了修改，在后端的 GPT回复日志(B文件夹) 中仍然保留真实的 diagnosis 过程供排查
            b_parts.append(f"[{idx}]\n{gpt_reply.strip()}")
        elif original_text == corrected_text:
            a_parts.append(f"[{idx}]\n<|same|>")
            # 【优化】同上
            b_parts.append(f"[{idx}]\n{gpt_reply.strip()}")
        else:
            # 文本确实发生了实质性改变，这下面的 readable_parts 逻辑会被完美触发
            a_parts.append(f"[{idx}]\n{original_text}")
            b_parts.append(f"[{idx}]\n{corrected_text}")

            img_tags = []
            img_filename = Path(image_path).name
            W, H = page_image.size

            if not seg.coords:
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
                    
                    MAX_WIDTH = 700
                    scale = min(1.0, MAX_WIDTH / w_px) if w_px > 0 else 1.0
                    zoom_style = f" zoom: {scale:.3f};" if scale < 1.0 else ""
                    
                    img_tags.append(
                        f'<img src="../images_pages/{img_filename}" style="width: {w_px}px; height: {h_px}px; object-fit: none; object-position: -{x1}px -{y1}px; max-width: none;{zoom_style}">'
                    )
            
            # 1. 构造原有 summary.md 表格
            def esc(txt: str) -> str:
                txt = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                return txt.replace("|", "&#124;").replace("\n", "<br>")
            
            summary_part = "\n".join(img_tags) + f"\n\n| Original | Modified |\n| --- | --- |\n| {esc(original_text)} | {esc(corrected_text)} |"
            summary_parts.append(summary_part)

            # 2. 构造【新增】易于观看的 readable_summary.md 片段
            readable_part = [
                f"### Segment [{idx}]",
                "#### User"
            ]
            readable_part.extend(img_tags)
            readable_part.append("```OCR\n" + original_text + "\n```")
            readable_part.append("#### Assistant")
            readable_part.append(gpt_reply)
            
            readable_parts.append("\n".join(readable_part))

    sep = "\n---\n"
    raw_a = sep.join(a_parts)
    raw_b = sep.join(b_parts)
    raw_response = sep.join(sorted(raw_parts))
    summary_md = "\n\n---\n\n".join(summary_parts) if summary_parts else ""
    readable_md = "\n\n---\n\n".join(readable_parts) if readable_parts else "" # <-- 【新增】

    return CorrectionResult(
        corrected=corrected,
        raw_response=raw_response,
        raw_a=raw_a,
        raw_b=raw_b,
        summary=summary_md,
        readable_summary=readable_md, # <-- 【新增】
        n_segments=len(segments),
        n_sent=len(non_image),
        n_ok=n_ok,
        n_image_skipped=n_image,
    )