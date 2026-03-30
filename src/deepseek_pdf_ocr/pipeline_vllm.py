# src/deepseek_pdf_ocr/pipeline_vllm.py
"""主 pipeline (vLLM 批量推理)：PDF → 批量 OCR → GPT 校正 → Markdown。
与 ``pipeline.py`` 功能完全一致，唯一区别在于 **Step 3 (DeepSeek OCR)**:
利用 ``ocr_api_server_vllm`` 的多图批量接口，将多张页面图像合并为一次
请求发送，大幅提升大规模 PDF 的吞吐量。
新增参数 ``ds_batch_size`` 控制每次 API 请求并行 OCR 的页数。
"""
from __future__ import annotations
import base64
import re
import time
from pathlib import Path
from typing import Sequence
from openai import OpenAI
from tqdm import tqdm
from deepseek_pdf_ocr.pdf_utils import (
    pdf_to_images,
    extract_text_from_pdf,
    get_page_count,
)
from deepseek_pdf_ocr.correction import run_gpt_correction
from deepseek_pdf_ocr.post_process import process_single_page
from deepseek_pdf_ocr.merge_markdown import merge_page_markdowns
from deepseek_pdf_ocr.pipeline import _format_duration, _print_timing_report

def _encode_image_b64(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

_IMAGE_MARKER_RE = re.compile(r"<!--\s*image\s+(\d+)\s*-->")

def _split_batch_response(text: str, expected: int) -> list[str]:
    # ... 保持原样 ...
    if expected <= 1:
        return [text.strip()]
    markers = list(_IMAGE_MARKER_RE.finditer(text))
    if not markers:
        return [text.strip()] + [""] * (expected - 1)
    segments: dict[int, str] = {}
    for i, m in enumerate(markers):
        idx = int(m.group(1))
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        chunk = text[start:end]
        chunk = re.sub(r"\s*---\s*$", "", chunk)
        segments[idx] = chunk.strip()
    return [segments.get(i, "") for i in range(1, expected + 1)]

from deepseek_pdf_ocr.ocr import DEFAULT_OCR_PROMPT

def _run_batch_ocr(
    page_nums: Sequence[int],
    images_dir: Path,
    ocr_dir: Path,
    *,
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
    prompt: str | None = None,
) -> None:
    # ... 保持原样 ...
    client = OpenAI(api_key=api_key, base_url=base_url)
    todo: list[int] = []
    for pn in page_nums:
        if (ocr_dir / f"page-{pn}.md").exists():
            print(f"  跳过第 {pn} 页 (已存在)")
        else:
            todo.append(pn)
    if not todo:
        print("  所有页面 OCR 结果已存在，跳过。")
        return
    n_batches = (len(todo) + batch_size - 1) // batch_size
    for bi in tqdm(
        range(0, len(todo), batch_size),
        desc="OCR批量处理",
        total=n_batches,
    ):
        batch = todo[bi : bi + batch_size]
        content: list[dict] = [
            {"type": "text", "text": prompt or DEFAULT_OCR_PROMPT},
        ]
        for pn in batch:
            b64 = _encode_image_b64(images_dir / f"{pn}.png")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
            )
            parts = _split_batch_response(
                resp.choices[0].message.content,
                len(batch),
            )
            for pn, txt in zip(batch, parts):
                (ocr_dir / f"page-{pn}.md").write_text(txt, encoding="utf-8")
                print(f"  ✓ 第 {pn} 页 OCR 完成")
        except Exception as exc:
            for pn in batch:
                print(f"  ✗ 第 {pn} 页 OCR 失败: {exc}")

def run_pipeline_vllm(
    pdf_path: str | Path,
    ds_api_key: str,
    ds_base_url: str,
    gpt_api_key: str,
    gpt_endpoint: str,
    *,
    dpi: int = 300,
    ds_model: str = "deepseek-ocr-2",
    ds_batch_size: int = 1,
    gpt_model: str = "gpt-5.4",
    gpt_temperature: float = 1.0,
    gpt_max_workers: int = 8,
    merge_markdown: bool = True,
    merged_filename: str = "ocr.md",
) -> Path:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
    timings: dict[str, float] = {}
    t_pipeline = time.perf_counter()

    base_dir = pdf_path.parent / pdf_path.stem
    images_dir = base_dir / "images_pages"
    text_dir = base_dir / "pdf_text"
    ocr_dir = base_dir / "deepseek-ocr-2"
    gpt_dir = base_dir / "gpt5.2"
    gpt_raw_dir = base_dir / "gpt5.2-raw"
    gpt_raw_a_dir = gpt_raw_dir / "A"   
    gpt_raw_b_dir = gpt_raw_dir / "B"   
    gpt_summary_pages_dir = gpt_raw_dir / "summary_pages"
    gpt_readable_pages_dir = gpt_raw_dir / "readable_pages" # <-- 【新增】易读版缓存文件夹
    output_dir = base_dir / "output"

    for d in (
        base_dir, images_dir, text_dir, ocr_dir,
        gpt_dir, gpt_raw_dir, gpt_raw_a_dir, gpt_raw_b_dir, gpt_summary_pages_dir,
        gpt_readable_pages_dir, # <-- 【新增】
        output_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
    num_pages = get_page_count(pdf_path)

    # ... [Step 1 到 Step 3 均保持不变] ...
    # ════════════ Step 1: PDF → 高清图像 ════════════
    print("=" * 60)
    print("Step 1: PDF 转高清图像")
    print("=" * 60)
    t0 = time.perf_counter()
    existing_images = list(images_dir.glob("*.png"))
    if len(existing_images) == num_pages:
        print(f"✓ 检测到已存在 {num_pages} 张图像，跳过转换。")
    else:
        num_pages = pdf_to_images(pdf_path, images_dir, dpi=dpi)
    timings["Step 1: PDF to Images"] = time.perf_counter() - t0

    # ════════════ Step 2: 提取 PDF 内嵌文本 ════════════
    print("\n" + "=" * 60)
    print("Step 2: 提取PDF内嵌文本")
    print("=" * 60)
    t0 = time.perf_counter()
    pdf_texts = extract_text_from_pdf(pdf_path)
    for pn, txt in pdf_texts.items():
        (text_dir / f"page-{pn}-text.txt").write_text(txt, encoding="utf-8")
    print(f"✓ 已提取 {len(pdf_texts)} 页文本")
    timings["Step 2: Extract PDF Text"] = time.perf_counter() - t0

    # ════════════ Step 3: DeepSeek OCR (vLLM batch) ════════════
    print("\n" + "=" * 60)
    print(f"Step 3: DeepSeek OCR-2  (vLLM, batch_size={ds_batch_size})")
    print("=" * 60)
    t0 = time.perf_counter()
    _run_batch_ocr(
        page_nums=list(range(1, num_pages + 1)),
        images_dir=images_dir,
        ocr_dir=ocr_dir,
        api_key=ds_api_key,
        base_url=ds_base_url,
        model=ds_model,
        batch_size=ds_batch_size,
    )
    timings["Step 3: DeepSeek OCR (vLLM)"] = time.perf_counter() - t0

    # ════════════ Step 4: GPT 校正 ════════════
    print("\n" + "=" * 60)
    print("Step 4: GPT 校正")
    print("=" * 60)
    t0 = time.perf_counter()
    for pn in tqdm(range(1, num_pages + 1), desc="GPT校正"):
        ocr_file = ocr_dir / f"page-{pn}.md"
        gpt_output = gpt_dir / f"page-{pn}.md"
        gpt_raw_a = gpt_raw_a_dir / f"page-{pn}.md"
        gpt_raw_b = gpt_raw_b_dir / f"page-{pn}.md"
        image_path = images_dir / f"{pn}.png"
        if gpt_output.exists():
            print(f"  跳过第 {pn} 页 (已存在)")
            continue
        if not ocr_file.exists():
            print(f"  跳过第 {pn} 页 (无OCR结果)")
            continue
        try:
            ocr_result = ocr_file.read_text(encoding="utf-8")
            extracted_text = pdf_texts.get(pn, "")
            result = run_gpt_correction(
                ocr_result,
                str(image_path),
                extracted_text,
                gpt_api_key,
                gpt_endpoint,
                model=gpt_model,
                temperature=gpt_temperature,
                max_workers=gpt_max_workers,
            )
            gpt_output.write_text(result.corrected, encoding="utf-8")
            gpt_raw_a.write_text(result.raw_a, encoding="utf-8")
            gpt_raw_b.write_text(result.raw_b, encoding="utf-8")
            
            (gpt_summary_pages_dir / f"page-{pn}.md").write_text(result.summary, encoding="utf-8")
            (gpt_readable_pages_dir / f"page-{pn}.md").write_text(result.readable_summary, encoding="utf-8") # <-- 【新增】写出按页易读文件
            
            print(
                f"  ✓ 第 {pn} 页 GPT 校正完成"
                f"  ({result.n_ok}/{result.n_sent} segments unchanged"
                f", {result.n_image_skipped} image skipped)"
            )
        except Exception as e:
            print(f"  ✗ 第 {pn} 页 GPT 校正失败: {e}")
            
    # ── Step 4.5 汇总所有的 Markdown 差异表格及易读版 ──
    summary_lines = ["# GPT Correction Summary\n"]
    readable_lines = ["# GPT Correction Readable Log\n"] # <-- 【新增】汇总易读版
    
    for pn in range(1, num_pages + 1):
        summary_lines.append(f"## Page {pn}\n")
        page_summary_file = gpt_summary_pages_dir / f"page-{pn}.md"
        if page_summary_file.exists():
            content = page_summary_file.read_text(encoding="utf-8").strip()
            if content:
                summary_lines.append(content + "\n")
            else:
                summary_lines.append("*No modifications*\n")
        else:
            summary_lines.append("*No summary available*\n")
        summary_lines.append("\n---\n")

        # 处理 readable 版本
        readable_lines.append(f"## Page {pn}\n")
        page_readable_file = gpt_readable_pages_dir / f"page-{pn}.md"
        if page_readable_file.exists():
            content_read = page_readable_file.read_text(encoding="utf-8").strip()
            if content_read:
                readable_lines.append(content_read + "\n")
            else:
                readable_lines.append("*No modifications*\n")
        else:
            readable_lines.append("*No readable log available*\n")
        readable_lines.append("\n---\n")

    (gpt_raw_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (gpt_raw_dir / "readable_summary.md").write_text("\n".join(readable_lines), encoding="utf-8") # <-- 【新增】写盘

    timings["Step 4: GPT Correction"] = time.perf_counter() - t0

    # ════════════ Step 5: 后处理 ════════════ (保持原样)
    print("\n" + "=" * 60)
    print("Step 5: 后处理 (提取图片、绘制边框)")
    print("=" * 60)
    t0 = time.perf_counter()
    for pn in range(1, num_pages + 1):
        page_out = output_dir / f"page-{pn}"
        if (
            (page_out / "result.md").exists()
            and (page_out / "result_with_boxes.jpg").exists()
        ):
            print(f"  跳过第 {pn} 页后处理 (结果已存在)")
            continue
        try:
            process_single_page(
                pn,
                input_dir=str(gpt_dir),
                output_dir=str(output_dir),
                image_dir=str(images_dir),
            )
            print(f"  ✓ 第 {pn} 页后处理完成")
        except Exception as e:
            print(f"  ✗ 第 {pn} 页后处理失败: {e}")
    timings["Step 5: Post-processing"] = time.perf_counter() - t0

    # ════════════ Step 6: 合并 Markdown ════════════ (保持原样)
    if merge_markdown:
        print("\n" + "=" * 60)
        print("Step 6: 合并所有页 Markdown")
        print("=" * 60)
        t0 = time.perf_counter()
        try:
            merged_md = merge_page_markdowns(
                output_dir, merged_filename=merged_filename,
            )
            print(f"✓ 已生成合并文件: {merged_md}")
        except Exception as e:
            print(f"✗ 合并 Markdown 失败: {e}")
        timings["Step 6: Merge Markdown"] = time.perf_counter() - t0
    total_time = time.perf_counter() - t_pipeline

    # ── Summary ──
    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)
    print(f"  PDF文件:        {pdf_path}")
    print(f"  工作目录:       {base_dir}")
    print(f"  页面图像:       {images_dir}")
    print(f"  PDF内嵌文本:    {text_dir}")
    print(f"  OCR原始结果:    {ocr_dir}")
    print(f"  GPT校正结果:    {gpt_dir}")
    print(f"  GPT原始回复 A:  {gpt_raw_a_dir}  (OCR原文，diff左侧)")
    print(f"  GPT原始回复 B:  {gpt_raw_b_dir}  (GPT回复，diff右侧)")
    print(f"  修改内容汇总:   {gpt_raw_dir / 'summary.md'}  (包含对比图片与前后文本)") 
    print(f"  易读格式汇总:   {gpt_raw_dir / 'readable_summary.md'}  (包含原始Prompt格式输出)") # <-- 【新增】控制台提示
    print(f"  最终输出:       {output_dir}")
    print(f"  OCR 批大小:     {ds_batch_size}")
    print(f"  GPT 并行数:     {gpt_max_workers}")
    print(f"\n每个页面的输出包括:")
    print(f"  - result.md:             处理后的markdown文件(带图片引用)")
    print(f"  - result_with_boxes.jpg: 带可视化边框的图片")
    print(f"  - images/:               提取的图片文件夹")
    print(f"\nVSCode diff 用法:")
    print(f"  在 gpt5.2-raw/ 目录下，A/ 为校正前原文，B/ 为GPT回复")
    print(f"  右键 A/page-N.md → 「选择以进行比较」，再右键 B/page-N.md → 「与已选项目比较」")
    _print_timing_report(timings, total_time)
    return output_dir