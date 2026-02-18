"""PDF 操作：转图像、提取内嵌文本。"""

import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm


def pdf_to_images(pdf_path: str | Path, output_dir: str | Path, dpi: int = 300) -> int:
    """将 PDF 每一页渲染为 PNG 图像。

    Parameters
    ----------
    pdf_path : path-like
        输入 PDF 文件路径。
    output_dir : path-like
        输出图像目录。
    dpi : int
        渲染分辨率，默认 300。

    Returns
    -------
    int
        总页数。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    num_pages = len(doc)
    print(f"PDF共 {num_pages} 页, DPI={dpi}")

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in tqdm(range(num_pages), desc="PDF转图像"):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image_path = output_dir / f"{page_num + 1}.png"
        pix.save(str(image_path))

    doc.close()
    print(f"✓ 已保存 {num_pages} 张图像到 {output_dir}")
    return num_pages


def extract_text_from_pdf(pdf_path: str | Path) -> dict[int, str]:
    """从 PDF 中按页提取内嵌文本。

    Returns
    -------
    dict[int, str]
        {页码(1-based): 文本}
    """
    doc = fitz.open(str(pdf_path))
    texts = {}
    for page_num in range(len(doc)):
        page = doc[page_num]
        texts[page_num + 1] = page.get_text("text")
    doc.close()
    return texts


def get_page_count(pdf_path: str | Path) -> int:
    """返回 PDF 总页数。"""
    doc = fitz.open(str(pdf_path))
    n = len(doc)
    doc.close()
    return n