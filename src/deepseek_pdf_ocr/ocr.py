"""DeepSeek OCR-2 调用。"""

import base64
from pathlib import Path
from openai import OpenAI


def _encode_image(image_path: str | Path) -> str:
    """将图像文件编码为 base64 字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_deepseek_ocr(
    image_path: str | Path,
    api_key: str,
    base_url: str,
    model: str = "deepseek-ocr-2",
) -> str:
    """使用 DeepSeek OCR-2 对单张图片进行 OCR。

    Parameters
    ----------
    image_path : path-like
        图片路径。
    api_key : str
        API Key。
    base_url : str
        API Base URL。
    model : str
        模型名称。

    Returns
    -------
    str
        OCR 识别出的 Markdown 文本。
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    image_base64 = _encode_image(image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Convert the document to markdown."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content