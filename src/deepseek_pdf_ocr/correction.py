"""GPT 校正：利用多模态 LLM 对 OCR 结果进行纠错。"""

import base64
from pathlib import Path
from openai import OpenAI


def _encode_image(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


_CORRECTION_PROMPT_TEMPLATE = r"""# Role
你是一位精通学术文档排版与 OCR 后处理的专家。

# Input Data
- **OCR 原始数据**(不含这一行): 

{ocr_result}

- **image**: 

(见附件图片)

- **PDF直接用代码提取出的文字**: 

{extracted_text}

# Goal
根据用户的指令，对 OCR 原始数据进行润色和纠错。重点是修复拼写错误、标点符号及数学公式语法，同时严格执行格式转换规则。

# Strict Constraints (必须严格遵守的准则)
1. **结构绝对保留**：
   - 严禁修改任何特殊标签（如 `<|ref|>`, `<|/ref|>`, `<|det|>`, `<|/det|>`）。
   - 严禁修改任何边界框坐标数据（即 `[[...]]` 内的数字）。
   - 保留 HTML 表格结构（`<table>`, `<tr>` 等）。
2. **LaTeX 格式转换 (Markdown 兼容)**：
   - **行内公式**：必须将原本的 `\( ... \)` 格式替换为 `$ ... $`。
   - **行间公式**：必须将原本的 `\[ ... \]` 格式替换为 `$$ ... $$`。
   - 确保公式内部语法正确。
   - 修正公式中的明显OCR错误
   - 确保公式中的上标 `^` 和下标 `_` 逻辑正确
3. **文本润色**：
   - 修正单词拼写错误（例如将 OCR 误识别的 `l` 修正为 `1`，或 `rn` 修正为 `m`，反之亦然）。
   - 确保标点符号使用正确

# Output Format
输出处理后的完整文本，保持原有的标签结构，但内容已优化且公式已转换为 Markdown 格式。"""


def run_gpt_correction(
    ocr_result: str,
    image_path: str | Path,
    extracted_text: str,
    api_key: str,
    endpoint: str,
    model: str = "gpt-5.2",
    temperature: float = 1.0,
) -> str:
    """使用多模态 LLM 对 OCR 结果进行校正。

    Parameters
    ----------
    ocr_result : str
        DeepSeek OCR 原始输出。
    image_path : path-like
        对应页面图片路径。
    extracted_text : str
        PDF 内嵌文本（辅助校正）。
    api_key : str
        API Key。
    endpoint : str
        API endpoint / base_url。
    model : str
        模型名称。
    temperature : float
        采样温度。

    Returns
    -------
    str
        校正后的 Markdown 文本。
    """
    client = OpenAI(base_url=endpoint, api_key=api_key)
    image_base64 = _encode_image(image_path)

    prompt = _CORRECTION_PROMPT_TEMPLATE.format(
        ocr_result=ocr_result,
        extracted_text=extracted_text,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content