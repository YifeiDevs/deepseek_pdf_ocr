"""加载 OCR 结果，组装为 OpenAI 多模态消息格式，供下游 LLM 使用。"""

from __future__ import annotations

import base64
import re
from pathlib import Path


def load_ocr_results_for_llm(
    pdf_path: str | Path,
    include_images: bool = False,
) -> list[dict]:
    """读取 pipeline 的 output 目录，返回 OpenAI content 列表。

    Parameters
    ----------
    pdf_path : path-like
        原始 PDF 路径（或其同名工作目录）。
    include_images : bool
        是否将 Markdown 中引用的图片以 base64 内联。

    Returns
    -------
    list[dict]
        可直接拼入 ``messages[].content`` 的列表。
    """
    pdf_path = Path(pdf_path)
    base_dir = pdf_path.parent / pdf_path.stem if pdf_path.is_file() else pdf_path
    output_dir = base_dir / "output"
    if not output_dir.exists():
        raise FileNotFoundError(f"找不到 output 目录: {output_dir}")

    page_dirs = sorted(
        [
            (int(d.name.split("-")[1]), d)
            for d in output_dir.iterdir()
            if d.is_dir() and d.name.startswith("page-")
        ]
    )

    llm_content: list[dict] = []
    img_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

    for page_num, page_dir in page_dirs:
        md_file = page_dir / "result.md"
        if not md_file.exists():
            continue

        text = md_file.read_text(encoding="utf-8")
        llm_content.append(
            {"type": "text", "text": f"\n\n--- Page {page_num} ---\n"}
        )

        if not include_images:
            # llm_content.append({"type": "text", "text": text})
            # 【核心修改点】: 此时不转 base64，但要修复相对路径为绝对路径
            # 这样 Agent (Langflow) 才能根据绝对路径在磁盘上找到图片
            
            def replace_with_abs_path(match):
                alt_text = match.group(1)
                rel_path = match.group(2)
                
                # 拼接绝对路径
                abs_img_path = (page_dir / rel_path).resolve()
                
                # 如果文件存在，替换为绝对路径；否则保留原样
                if abs_img_path.exists():
                    # 注意：在 Windows 上 str(Path) 会生成反斜杠，通常 Markdown 解析器能处理
                    # 为了兼容性最好，也可以使用 .as_posix()，但 Langflow 本地加载可能更喜欢系统原声路径
                    return f"![{alt_text}]({str(abs_img_path)})"
                return match.group(0)

            # 使用正则 sub 函数进行全文替换
            text_with_abs_paths = img_pattern.sub(replace_with_abs_path, text)
            llm_content.append({"type": "text", "text": text_with_abs_paths})
        else:
            last_end = 0
            for m in img_pattern.finditer(text):
                before = text[last_end : m.start()].strip()
                if before:
                    llm_content.append({"type": "text", "text": text[last_end : m.start()]})

                img_path = page_dir / m.group(2)
                if img_path.exists():
                    mime = (
                        "image/png"
                        if img_path.suffix.lower() == ".png"
                        else "image/jpeg"
                    )
                    b64 = base64.b64encode(img_path.read_bytes()).decode()
                    llm_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        }
                    )
                else:
                    llm_content.append({"type": "text", "text": m.group(0)})
                last_end = m.end()

            trailing = text[last_end:].strip()
            if trailing:
                llm_content.append({"type": "text", "text": text[last_end:]})

    return llm_content