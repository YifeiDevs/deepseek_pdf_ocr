"""合并每页 Markdown 输出为单个 Markdown 文件。"""

from __future__ import annotations

import re
from pathlib import Path


_IMG_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")


def merge_page_markdowns(
    output_dir: str | Path,
    merged_filename: str = "merged.md",
) -> Path:
    """合并 ``output/page-N/result.md`` 为单个 Markdown。

    合并后的文件写入 ``output`` 的父目录（即工作目录根）下的 ``{merged_filename}``。

    关键点：每页的图片都在各自的 ``output/page-N/images/`` 下，
    因此需要把每页 Markdown 里的 ``images/...`` 改写为 ``output/page-N/images/...``。

    Parameters
    ----------
    output_dir : path-like
        pipeline 的 ``output`` 目录。
    merged_filename : str
        合并后的文件名。

    Returns
    -------
    Path
        合并后的 Markdown 文件路径。
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"output 目录不存在: {output_dir}")

    base_dir = output_dir.parent

    page_dirs: list[tuple[int, Path]] = []
    for d in output_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("page-"):
            continue
        try:
            page_num = int(d.name.split("-", 1)[1])
        except ValueError:
            continue
        page_dirs.append((page_num, d))
    page_dirs.sort(key=lambda x: x[0])

    merged_path = base_dir / merged_filename
    parts: list[str] = []

    for page_num, page_dir in page_dirs:
        md_path = page_dir / "result.md"
        if not md_path.exists():
            continue

        text = md_path.read_text(encoding="utf-8")

        def _rewrite_img(match: re.Match[str]) -> str:
            alt = match.group("alt")
            raw_path = match.group("path").strip()

            # 仅改写相对路径（避免误改 http(s) / data: / 绝对路径）
            lowered = raw_path.lower()
            if lowered.startswith(("http://", "https://", "data:")):
                return match.group(0)
            if Path(raw_path).is_absolute():
                return match.group(0)

            # 每页 result.md 默认引用 images/...
            # 合并到 base_dir/merged.md 后，需要变成 output/page-N/images/...
            if raw_path.startswith("images/") or raw_path.startswith("images\\"):
                rel = raw_path.replace("\\", "/")
                new_path = f"output/page-{page_num}/{rel}"
                return f"![{alt}]({new_path})"

            # 其他相对路径：保守处理，保持原样
            return match.group(0)

        text = _IMG_PATTERN.sub(_rewrite_img, text)

        parts.append(f"\n\n---\n\n## Page {page_num}\n\n")
        parts.append(text.rstrip() + "\n")

    merged_path.write_text("".join(parts).lstrip(), encoding="utf-8")
    return merged_path
