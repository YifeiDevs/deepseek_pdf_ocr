"""主 pipeline：串联 PDF 解析 → OCR → 校正 → 后处理全流程。"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from deepseek_pdf_ocr.pdf_utils import pdf_to_images, extract_text_from_pdf, get_page_count
from deepseek_pdf_ocr.ocr import run_deepseek_ocr
from deepseek_pdf_ocr.correction import run_gpt_correction
from deepseek_pdf_ocr.post_process import process_single_page


def run_pipeline(
    pdf_path: str | Path,
    ds_api_key: str,
    ds_base_url: str,
    gpt_api_key: str,
    gpt_endpoint: str,
    *,
    dpi: int = 300,
    ds_model: str = "deepseek-ocr-2",
    gpt_model: str = "gpt-5.2",
    gpt_temperature: float = 1.0,
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

    Returns
    -------
    Path
        输出根目录。
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    # ── 目录结构 ──
    base_dir = pdf_path.parent / pdf_path.stem
    images_dir = base_dir / "images_pages"
    text_dir = base_dir / "pdf_text"
    ocr_dir = base_dir / "deepseek-ocr-2"
    gpt_dir = base_dir / "gpt5.2"
    output_dir = base_dir / "output"

    for d in [base_dir, images_dir, text_dir, ocr_dir, gpt_dir, output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    num_pages = get_page_count(pdf_path)

    # ── Step 1: PDF → 高清图像 ──
    print("=" * 60)
    print("Step 1: PDF 转高清图像")
    print("=" * 60)
    existing_images = list(images_dir.glob("*.png"))
    if len(existing_images) == num_pages:
        print(f"✓ 检测到已存在 {num_pages} 张图像，跳过转换。")
    else:
        num_pages = pdf_to_images(pdf_path, images_dir, dpi=dpi)

    # ── Step 2: 提取 PDF 文本 ──
    print("\n" + "=" * 60)
    print("Step 2: 提取PDF内嵌文本")
    print("=" * 60)
    pdf_texts = extract_text_from_pdf(pdf_path)
    for page_num, text in pdf_texts.items():
        text_file = text_dir / f"page-{page_num}-text.txt"
        text_file.write_text(text, encoding="utf-8")
    print(f"✓ 已提取 {len(pdf_texts)} 页文本")

    # ── Step 3: DeepSeek OCR ──
    print("\n" + "=" * 60)
    print("Step 3: DeepSeek OCR-2")
    print("=" * 60)
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

    # ── Step 4: GPT 校正 ──
    print("\n" + "=" * 60)
    print("Step 4: GPT 校正")
    print("=" * 60)
    for page_num in tqdm(range(1, num_pages + 1), desc="GPT校正"):
        ocr_file = ocr_dir / f"page-{page_num}.md"
        gpt_output = gpt_dir / f"page-{page_num}.md"
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
            gpt_output.write_text(gpt_result, encoding="utf-8")
            print(f"  ✓ 第 {page_num} 页 GPT 校正完成")
        except Exception as e:
            print(f"  ✗ 第 {page_num} 页 GPT 校正失败: {e}")

    # ── Step 5: 后处理 ──
    print("\n" + "=" * 60)
    print("Step 5: 后处理 (提取图片、绘制边框)")
    print("=" * 60)
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
    print(f"  最终输出:       {output_dir}")
    print(f"\n每个页面的输出包括:")
    print(f"  - result.md:             处理后的markdown文件(带图片引用)")
    print(f"  - result_with_boxes.jpg: 带可视化边框的图片")
    print(f"  - images/:               提取的图片文件夹")

    return output_dir