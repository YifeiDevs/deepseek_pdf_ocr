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

# ── Reuse utilities from the original pipeline & sister modules ──
from deepseek_pdf_ocr.pdf_utils import (
    pdf_to_images,
    extract_text_from_pdf,
    get_page_count,
)
from deepseek_pdf_ocr.correction import run_gpt_correction
from deepseek_pdf_ocr.post_process import process_single_page
from deepseek_pdf_ocr.merge_markdown import merge_page_markdowns
from deepseek_pdf_ocr.pipeline import _format_duration, _print_timing_report


# ╔══════════════════════════════════════════════════════════════╗
# ║  Internal helpers                                            ║
# ╚══════════════════════════════════════════════════════════════╝


def _encode_image_b64(path: str | Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Matches ``<!-- image N -->`` headers emitted by ``ocr_api_server_vllm``
_IMAGE_MARKER_RE = re.compile(r"<!--\s*image\s+(\d+)\s*-->")


def _split_batch_response(text: str, expected: int) -> list[str]:
    """Split a combined multi-image response into *expected* individual results.

    ``ocr_api_server_vllm`` formats multi-image results as::

        <!-- image 1 -->
        …per-page OCR content…

        ---

        <!-- image 2 -->
        …per-page OCR content…

    Single-image responses contain **no** markers and are returned as-is.

    Parameters
    ----------
    text : str
        Raw ``response.choices[0].message.content``.
    expected : int
        Number of images that were sent in the batch.

    Returns
    -------
    list[str]
        A list of length *expected*; missing entries are empty strings.
    """
    if expected <= 1:
        return [text.strip()]

    markers = list(_IMAGE_MARKER_RE.finditer(text))
    if not markers:
        # Fallback: server returned plain text without markers
        return [text.strip()] + [""] * (expected - 1)

    # Build {1-based image index → content} mapping
    segments: dict[int, str] = {}
    for i, m in enumerate(markers):
        idx = int(m.group(1))
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        chunk = text[start:end]
        # Strip the trailing ``\n\n---\n\n`` separator between images
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
    """OCR pages in batches via the vLLM OpenAI-compatible endpoint.

    Pages whose output (``ocr_dir/page-{N}.md``) already exists are
    automatically skipped.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    # ── filter out already-completed pages ──
    todo: list[int] = []
    for pn in page_nums:
        if (ocr_dir / f"page-{pn}.md").exists():
            print(f"  跳过第 {pn} 页 (已存在)")
        else:
            todo.append(pn)

    if not todo:
        print("  所有页面 OCR 结果已存在，跳过。")
        return

    # ── send images in batches ──
    n_batches = (len(todo) + batch_size - 1) // batch_size
    for bi in tqdm(
        range(0, len(todo), batch_size),
        desc="OCR批量处理",
        total=n_batches,
    ):
        batch = todo[bi : bi + batch_size]

        # Build multi-image content for the OpenAI-style API
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


# ╔══════════════════════════════════════════════════════════════╗
# ║  Public API                                                  ║
# ╚══════════════════════════════════════════════════════════════╝


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
    gpt_model: str = "gpt-5.2",
    gpt_temperature: float = 1.0,
    merge_markdown: bool = True,
    merged_filename: str = "ocr.md",
) -> Path:
    """Execute the full PDF OCR pipeline using the vLLM batch backend.

    API-compatible with :func:`pipeline.run_pipeline`; the only addition
    is *ds_batch_size*.

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
    ds_batch_size : int
        **每次 OCR 请求并行处理的页数**。

        - ``1`` — 逐页发送，行为与 ``pipeline.run_pipeline`` 完全相同。
        - ``> 1`` — 多张图片打包为一个请求，由 vLLM 引擎批量推理后
          拆分回各页结果。值越大吞吐越高，但单次请求 payload 与延迟
          也越大，请根据 GPU 显存和网络情况调整。
    gpt_model : str
        GPT 校正模型名称。
    gpt_temperature : float
        GPT 采样温度。
    merge_markdown : bool
        是否在 pipeline 结束后合并所有页的 result.md。
    merged_filename : str
        合并后的 Markdown 文件名（写入工作目录根）。

    Returns
    -------
    Path
        输出根目录 (``output_dir``)。
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    timings: dict[str, float] = {}
    t_pipeline = time.perf_counter()

    # ── Directory layout (identical to pipeline.py) ──
    base_dir = pdf_path.parent / pdf_path.stem
    images_dir = base_dir / "images_pages"
    text_dir = base_dir / "pdf_text"
    ocr_dir = base_dir / "deepseek-ocr-2"
    gpt_dir = base_dir / "gpt5.2"
    gpt_raw_dir = base_dir / "gpt5.2-raw"
    output_dir = base_dir / "output"

    for d in (base_dir, images_dir, text_dir, ocr_dir, gpt_dir, gpt_raw_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    num_pages = get_page_count(pdf_path)

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
        gpt_raw = gpt_raw_dir / f"page-{pn}.md"
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
            )
            gpt_output.write_text(result.corrected, encoding="utf-8")
            gpt_raw.write_text(result.raw_response, encoding="utf-8")
            print(
                f"  ✓ 第 {pn} 页 GPT 校正完成"
                f"  ({result.n_ok}/{result.n_sent} segments unchanged"
                f", {result.n_image_skipped} image skipped)"
            )
        except Exception as e:
            print(f"  ✗ 第 {pn} 页 GPT 校正失败: {e}")
    timings["Step 4: GPT Correction"] = time.perf_counter() - t0

    # ════════════ Step 5: 后处理 ════════════
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

    # ════════════ Step 6: 合并 Markdown ════════════
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
    print(f"  最终输出:       {output_dir}")
    print(f"  OCR 批大小:     {ds_batch_size}")
    print(f"\n每个页面的输出包括:")
    print(f"  - result.md:             处理后的markdown文件(带图片引用)")
    print(f"  - result_with_boxes.jpg: 带可视化边框的图片")
    print(f"  - images/:               提取的图片文件夹")

    _print_timing_report(timings, total_time)

    return output_dir