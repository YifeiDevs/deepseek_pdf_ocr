"""主 pipeline：串联 PDF 解析 → OCR → 校正 → 后处理全流程。"""
from __future__ import annotations
import time
from pathlib import Path
from tqdm import tqdm
from deepseek_pdf_ocr.pdf_utils import pdf_to_images, extract_text_from_pdf, get_page_count
from deepseek_pdf_ocr.ocr import run_deepseek_ocr
from deepseek_pdf_ocr.correction import run_gpt_correction
from deepseek_pdf_ocr.post_process import process_single_page
from deepseek_pdf_ocr.merge_markdown import merge_page_markdowns
def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.2f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.2f}s"
def _print_timing_report(timings: dict[str, float], total_time: float) -> None:
    """Print a nicely aligned timing report after the pipeline finishes."""
    print("\n" + "=" * 70)
    print("  Timing Report")
    print("=" * 70)
    name_width = max(len(name) for name in timings)
    bar_total = 40
    for name, duration in timings.items():
        pct = (duration / total_time * 100) if total_time > 0 else 0
        filled = int(pct / 100 * bar_total)
        bar = "█" * filled + "░" * (bar_total - filled)
        print(
            f"  {name:<{name_width}}"
            f"  {_format_duration(duration):>12}"
            f"  {pct:5.1f}%"
            f"  {bar}"
        )
    separator_width = name_width + 12 + 5 + bar_total + 10
    print(f"  {'─' * separator_width}")
    print(
        f"  {'Total':<{name_width}}"
        f"  {_format_duration(total_time):>12}"
        f"  100.0%"
    )
    print("=" * 70)
def run_pipeline(
    pdf_path: str | Path,
    ds_api_key: str,
    ds_base_url: str,
    gpt_api_key: str,
    gpt_endpoint: str,
    *,
    dpi: int = 300,
    ds_model: str = "deepseek-ocr-2",
    gpt_model: str = "gpt-5.4",
    gpt_temperature: float = 1.0,
    merge_markdown: bool = True,
    merged_filename: str = "ocr.md",
) -> Path:
    """执行完整的 PDF OCR pipeline。
    Parameters
    ----------
    pdf_path : path-like
        输入 PDF 文件路径。
    ds_api_key : str
        DeepSeek OCR API Key。
    ds_base_url : str
        DeepSeek OCR API base URL。
    gpt_api_key : str
        GPT 校正 API Key。
    gpt_endpoint : str
        GPT 校正 API endpoint。
    dpi : int
        PDF 渲染 DPI。
    ds_model : str
        DeepSeek OCR 模型名称。
    gpt_model : str
        GPT 校正模型名称。
    gpt_temperature : float
        GPT 采样温度。
    merge_markdown : bool
        是否在 pipeline 结束后合并所有页的 result.md。
    merged_filename : str
        合并后的 Markdown 文件名（写入工作目录根，即 output 的父目录）。
    Returns
    -------
    Path
        输出根目录。
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
    # ── Timers ──
    timings: dict[str, float] = {}
    pipeline_start = time.perf_counter()
    # ── 目录结构 ──
    base_dir = pdf_path.parent / pdf_path.stem
    images_dir = base_dir / "images_pages"
    text_dir = base_dir / "pdf_text"
    ocr_dir = base_dir / "deepseek-ocr-2"
    gpt_dir = base_dir / "gpt5.2"
    gpt_raw_dir = base_dir / "gpt5.2-raw"
    # A/B subdirectories for side-by-side diff in VSCode
    gpt_raw_a_dir = gpt_raw_dir / "A"   # OCR 原文（校正前）
    gpt_raw_b_dir = gpt_raw_dir / "B"   # GPT 回复（校正内容）
    output_dir = base_dir / "output"
    for d in [
        base_dir, images_dir, text_dir, ocr_dir,
        gpt_dir, gpt_raw_dir, gpt_raw_a_dir, gpt_raw_b_dir,
        output_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    num_pages = get_page_count(pdf_path)
    # ── Step 1: PDF → 高清图像 ──
    print("=" * 60)
    print("Step 1: PDF 转高清图像")
    print("=" * 60)
    step_start = time.perf_counter()
    existing_images = list(images_dir.glob("*.png"))
    if len(existing_images) == num_pages:
        print(f"✓ 检测到已存在 {num_pages} 张图像，跳过转换。")
    else:
        num_pages = pdf_to_images(pdf_path, images_dir, dpi=dpi)
    timings["Step 1: PDF to Images"] = time.perf_counter() - step_start
    # ── Step 2: 提取 PDF 文本 ──
    print("\n" + "=" * 60)
    print("Step 2: 提取PDF内嵌文本")
    print("=" * 60)
    step_start = time.perf_counter()
    pdf_texts = extract_text_from_pdf(pdf_path)
    for page_num, text in pdf_texts.items():
        text_file = text_dir / f"page-{page_num}-text.txt"
        text_file.write_text(text, encoding="utf-8")
    print(f"✓ 已提取 {len(pdf_texts)} 页文本")
    timings["Step 2: Extract PDF Text"] = time.perf_counter() - step_start
    # ── Step 3: DeepSeek OCR ──
    print("\n" + "=" * 60)
    print("Step 3: DeepSeek OCR-2")
    print("=" * 60)
    step_start = time.perf_counter()
    for page_num in tqdm(range(1, num_pages + 1), desc="OCR处理"):
        image_path = images_dir / f"{page_num}.png"
        ocr_output = ocr_dir / f"page-{page_num}.md"
        if ocr_output.exists():
            print(f"  跳过第 {page_num} 页 (已存在)")
            continue
        try:
            ocr_result = run_deepseek_ocr(
                str(image_path), ds_api_key, ds_base_url, model=ds_model,
            )
            ocr_output.write_text(ocr_result, encoding="utf-8")
            print(f"  ✓ 第 {page_num} 页 OCR 完成")
        except Exception as e:
            print(f"  ✗ 第 {page_num} 页 OCR 失败: {e}")
    timings["Step 3: DeepSeek OCR"] = time.perf_counter() - step_start
    # ── Step 4: GPT 校正 ──
    print("\n" + "=" * 60)
    print("Step 4: GPT 校正")
    print("=" * 60)
    step_start = time.perf_counter()
    for page_num in tqdm(range(1, num_pages + 1), desc="GPT校正"):
        ocr_file = ocr_dir / f"page-{page_num}.md"
        gpt_output = gpt_dir / f"page-{page_num}.md"
        # A = OCR 原文（校正前），B = GPT 回复（校正内容/理由）
        gpt_raw_a = gpt_raw_a_dir / f"page-{page_num}.md"
        gpt_raw_b = gpt_raw_b_dir / f"page-{page_num}.md"
        image_path = images_dir / f"{page_num}.png"
        if gpt_output.exists():
            print(f"  跳过第 {page_num} 页 (已存在)")
            continue
        if not ocr_file.exists():
            print(f"  跳过第 {page_num} 页 (无OCR结果)")
            continue
        try:
            ocr_result = ocr_file.read_text(encoding="utf-8")
            extracted_text = pdf_texts.get(page_num, "")
            gpt_result = run_gpt_correction(
                ocr_result,
                str(image_path),
                extracted_text,
                gpt_api_key,
                gpt_endpoint,
                model=gpt_model,
                temperature=gpt_temperature,
            )
            gpt_output.write_text(gpt_result.corrected, encoding="utf-8")
            # A: 有变化的 segment 写原文，<|ok|> 的写 <|ok|>（diff 左侧）
            gpt_raw_a.write_text(gpt_result.raw_a, encoding="utf-8")
            # B: 有变化的 segment 写 GPT 回复，<|ok|> 的写 <|ok|>（diff 右侧）
            gpt_raw_b.write_text(gpt_result.raw_b, encoding="utf-8")
            print(f"  ✓ 第 {page_num} 页 GPT 校正完成")
        except Exception as e:
            print(f"  ✗ 第 {page_num} 页 GPT 校正失败: {e}")
    timings["Step 4: GPT Correction"] = time.perf_counter() - step_start
    # ── Step 5: 后处理 ──
    print("\n" + "=" * 60)
    print("Step 5: 后处理 (提取图片、绘制边框)")
    print("=" * 60)
    step_start = time.perf_counter()
    for page_num in range(1, num_pages + 1):
        page_out = output_dir / f"page-{page_num}"
        if (
            (page_out / "result.md").exists()
            and (page_out / "result_with_boxes.jpg").exists()
        ):
            print(f"  跳过第 {page_num} 页后处理 (结果已存在)")
            continue
        try:
            process_single_page(
                page_num,
                input_dir=str(gpt_dir),
                output_dir=str(output_dir),
                image_dir=str(images_dir),
            )
            print(f"  ✓ 第 {page_num} 页后处理完成")
        except Exception as e:
            print(f"  ✗ 第 {page_num} 页后处理失败: {e}")
    timings["Step 5: Post-processing"] = time.perf_counter() - step_start
    if merge_markdown:
        # ── Step 6: 合并所有页 Markdown ──
        print("\n" + "=" * 60)
        print("Step 6: 合并所有页 Markdown")
        print("=" * 60)
        step_start = time.perf_counter()
        try:
            merged_md = merge_page_markdowns(output_dir, merged_filename=merged_filename)
            print(f"✓ 已生成合并文件: {merged_md}")
        except Exception as e:
            print(f"✗ 合并 Markdown 失败: {e}")
        timings["Step 6: Merge Markdown"] = time.perf_counter() - step_start
    total_time = time.perf_counter() - pipeline_start
    # ── 完成 ──
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
    print(f"  最终输出:       {output_dir}")
    print(f"\n每个页面的输出包括:")
    print(f"  - result.md:             处理后的markdown文件(带图片引用)")
    print(f"  - result_with_boxes.jpg: 带可视化边框的图片")
    print(f"  - images/:               提取的图片文件夹")
    print(f"\nVSCode diff 用法:")
    print(f"  在 gpt5.2-raw/ 目录下，A/ 为校正前原文，B/ 为GPT回复")
    print(f"  右键 A/page-N.md → 「选择以进行比较」，再右键 B/page-N.md → 「与已选项目比较」")
    # ── Timing Report ──
    _print_timing_report(timings, total_time)
    return output_dir