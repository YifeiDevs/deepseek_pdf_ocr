"""合并每页 Markdown 输出为单个 Markdown 文件。"""

from __future__ import annotations

import re
from pathlib import Path

_IMG_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")


def merge_page_markdowns(
    output_dir: str | Path,
    merged_filename: str = "merged.md",
) -> Path:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"output 目录不存在: {output_dir}")

    base_dir = output_dir.parent
    page_dirs: list[tuple[int, Path]] = []
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("page-"):
            try:
                page_dirs.append((int(d.name.split("-", 1)[1]), d))
            except ValueError:
                pass
    page_dirs.sort(key=lambda x: x[0])

    parts: list[str] = []
    for page_num, page_dir in page_dirs:
        md_path = page_dir / "result.md"
        if not md_path.exists():
            continue

        def _rewrite_img(match: re.Match[str], _pn: int = page_num) -> str:
            alt, raw = match.group("alt"), match.group("path").strip()
            if raw.lower().startswith(("http://", "https://", "data:")) or Path(raw).is_absolute():
                return match.group(0)
            if raw.replace("\\", "/").startswith("images/"):
                return f"![{alt}](output/page-{_pn}/{raw.replace(chr(92), '/')})"
            return match.group(0)

        text = _IMG_PATTERN.sub(_rewrite_img, md_path.read_text(encoding="utf-8"))
        sep = "\n\n---\n\n" if parts else ""
        parts.append(f"{sep}## Page {page_num}\n\n{text.rstrip()}\n")

    merged_path = base_dir / merged_filename
    merged_path.write_text("".join(parts), encoding="utf-8")
    return merged_path