"""后处理：解析 ref/det 标记、绘制可视化边框、提取嵌入图片。"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# 僵尸代码保留标记: torch 在原始脚本中被 import 但未使用
import torch  # noqa: F401 — kept intentionally (zombie code)


# ---------------------------------------------------------------------------
# ref 标记解析
# ---------------------------------------------------------------------------

def re_match(text: str):
    """提取文本中的 ``<|ref|>...<|/ref|><|det|>...<|/det|>`` 标记。

    Returns
    -------
    tuple
        (matches_all, matches_image, matches_other)
        每个 match 为 ``(full_str, label, coords_str)`` 三元组。
    """
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image: list[str] = []
    matches_other: list[str] = []
    for a_match in matches:
        if "<|ref|>image<|/ref|>" in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])

    return matches, matches_image, matches_other


def extract_coordinates_and_label(
    ref_text: tuple,
    image_width: int,
    image_height: int,
) -> tuple[str, list] | None:
    """从 ref 三元组中提取标签与坐标列表。"""
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])  # noqa: S307
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return None
    return (label_type, cor_list)


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _load_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """尝试加载 TrueType 字体，失败则回退到默认字体。"""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_bounding_boxes(
    image: Image.Image,
    refs: list[tuple],
    output_path: str | Path,
) -> Image.Image:
    """在图片上绘制边框并裁剪 image 类型区域。

    Parameters
    ----------
    image : PIL.Image
        原始页面图片。
    refs : list
        ``re_match`` 返回的 matches_all 列表。
    output_path : path-like
        页面输出目录（裁剪的图片保存到 ``output_path/images/``）。

    Returns
    -------
    PIL.Image
        绘制了边框的图片副本。
    """
    output_path = Path(output_path)
    image_width, image_height = image.size

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    font = _load_font(20)
    img_idx = 0

    for _i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result is None:
                continue

            label_type, points_list = result

            # 随机颜色
            color = (
                np.random.randint(0, 200),
                np.random.randint(0, 200),
                np.random.randint(0, 255),
            )
            color_a = color + (20,)

            for points in points_list:
                x1, y1, x2, y2 = points
                # 归一化坐标 → 像素坐标
                x1 = int(x1 / 999 * image_width)
                y1 = int(y1 / 999 * image_height)
                x2 = int(x2 / 999 * image_width)
                y2 = int(y2 / 999 * image_height)

                # 裁剪 image 区域
                if label_type == "image":
                    try:
                        cropped = image.crop((x1, y1, x2, y2))
                        cropped.save(output_path / "images" / f"{img_idx}.jpg")
                    except Exception as e:
                        print(f"Error cropping image: {e}")
                    img_idx += 1

                # 绘制边框
                try:
                    width = 4 if label_type == "title" else 2
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                    draw2.rectangle(
                        [x1, y1, x2, y2],
                        fill=color_a,
                        outline=(0, 0, 0, 0),
                        width=1,
                    )

                    # 标签
                    text_x = x1
                    text_y = max(0, y1 - 15)
                    text_bbox = draw.textbbox((0, 0), label_type, font=font)
                    tw = text_bbox[2] - text_bbox[0]
                    th = text_bbox[3] - text_bbox[1]
                    draw.rectangle(
                        [text_x, text_y, text_x + tw, text_y + th],
                        fill=(255, 255, 255, 30),
                    )
                    draw.text((text_x, text_y), label_type, font=font, fill=color)
                except Exception:
                    pass
        except Exception:
            continue

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


# ---------------------------------------------------------------------------
# 单页后处理
# ---------------------------------------------------------------------------

def process_single_page(
    page_num: int,
    input_dir: str | Path,
    output_dir: str | Path,
    image_dir: str | Path,
) -> None:
    """对单个页面执行完整后处理。

    1. 读取 GPT 校正后的 Markdown
    2. 替换 image ref 为 ``![](images/N.jpg)``
    3. 清理特殊字符
    4. 保存处理后 Markdown + 带边框可视化图片

    Parameters
    ----------
    page_num : int
        页码 (1-based)。
    input_dir : path-like
        GPT 校正结果目录（含 ``page-{N}.md``）。
    output_dir : path-like
        输出根目录。
    image_dir : path-like
        页面 PNG 图片目录。
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    image_dir = Path(image_dir)

    # 读取 Markdown
    md_file = input_dir / f"page-{page_num}.md"
    if not md_file.exists():
        print(f"Warning: {md_file} not found")
        return

    content = md_file.read_text(encoding="utf-8")

    # 读取对应图片
    image_file = image_dir / f"{page_num}.png"
    if not image_file.exists():
        print(f"Warning: {image_file} not found")
        return

    image = Image.open(image_file).convert("RGB")

    # 创建输出目录
    page_output_dir = output_dir / f"page-{page_num}"
    page_output_dir.mkdir(parents=True, exist_ok=True)
    (page_output_dir / "images").mkdir(exist_ok=True)

    # 提取 ref 标记
    matches_ref, matches_images, matches_other = re_match(content)

    # 替换图片引用
    processed_content = content
    for idx, a_match_image in enumerate(
        tqdm(matches_images, desc=f"Processing images for page {page_num}")
    ):
        processed_content = processed_content.replace(
            a_match_image,
            f"![](images/{idx}.jpg)\n",
        )

    # # 移除其他ref标记  — 僵尸代码，按要求保留
    # for idx, a_match_other in enumerate(matches_other):
    #     processed_content = processed_content.replace(a_match_other, '')

    # 清理特殊字符
    processed_content = processed_content.replace("\\coloneqq", ":=").replace(
        "\\eqqcolon", "=:"
    )

    # 保存 Markdown
    (page_output_dir / "result.md").write_text(processed_content, encoding="utf-8")

    # 绘制带边框的图片
    if matches_ref:
        result_image = draw_bounding_boxes(image, matches_ref, page_output_dir)
        result_image.save(page_output_dir / "result_with_boxes.jpg")

    print(f"✓ Processed page {page_num}")