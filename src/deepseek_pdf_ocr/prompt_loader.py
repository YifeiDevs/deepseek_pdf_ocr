from __future__ import annotations

import base64,json,mimetypes
from copy import deepcopy
from pathlib import Path

from markdown_it import MarkdownIt


def load_markdown_messages(path: str | Path, dump_json: str | Path | None = None) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []

    src = path.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tokens = MarkdownIt("commonmark").parse(src)
    content, i = [], 0

    def add_text(text: str):
        text = text.strip()
        if not text:
            return
        if content and content[-1]["type"] == "text":
            content[-1]["text"] += "\n\n" + text
        else:
            content.append({"type": "text", "text": text})

    while i < len(tokens):
        t = tokens[i]

        if t.type in {"paragraph_open", "heading_open"} and i + 1 < len(tokens) and tokens[i + 1].type == "inline":
            inline = tokens[i + 1]
            img = next((c.attrGet("src") for c in (inline.children or []) if c.type == "image"), None)

            if img:
                if img.startswith(("http://", "https://")):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img},
                    })
                else:
                    p = (path.parent / img).resolve()
                    if p.exists():
                        mime = mimetypes.guess_type(p)[0] or "image/png"
                        b64 = base64.b64encode(p.read_bytes()).decode()
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        })
                    else:
                        add_text("".join(lines[slice(*(t.map or (0, 0)))]))
            else:
                add_text("".join(lines[slice(*(t.map or (0, 0)))]))

            i += 3
            continue

        if t.type in {"fence", "code_block", "html_block"}:
            add_text("".join(lines[slice(*(t.map or (0, 0)))]))
            i += 1
            continue

        i += 1

    messages = [{"role": "user", "content": content}] if content else []

    if dump_json:
        Path(dump_json).write_text(
            json.dumps(messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return messages

def print_messages(msgs):
    msgs = deepcopy(msgs)
    for m in msgs:
        for x in m.get("content", []):
            u = x.get("image_url", {}).get("url", "")
            h, s, t = u.partition("base64,")
            if x.get("type") == "image_url" and s:
                x["image_url"]["url"] = f"{h}{s}{t[:4]}..."
    d = lambda x: json.dumps(x, ensure_ascii=False, separators=(", ", ": "))
    out = []
    for m in msgs:
        c = m.get("content", [])
        w = max((len(d(x.get("type"))) for x in c), default=0)
        rows = []
        for x in c:
            t = d(x["type"])
            k = next(k for k in x if k != "type")
            rows.append(f'    {{"type": {t},{(w - len(t) + 1) * " "}{d(k)}: {d(x[k])}}}')
        out.append(f'{{"role": {d(m.get("role"))},"content": [\n' + ",\n".join(rows) + "\n]}")
    print("[" + ",\n ".join(out) + "]")

if __name__ == "__main__":
    md_path = (Path(__file__).resolve().parent / "../../examples.md").resolve()
    msgs = load_markdown_messages(md_path, dump_json="examples.debug.json")
    print_messages(msgs)