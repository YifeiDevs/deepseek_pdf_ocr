"""命令行入口。"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="deepseek-pdf-ocr",
        description="PDF OCR pipeline: PDF → DeepSeek OCR → GPT correction → Markdown",
    )
    parser.add_argument("pdf_path", help="输入 PDF 文件路径")
    parser.add_argument("--ds-api-key", default="NVIDIA RTX A6000", help="DeepSeek OCR API Key")
    parser.add_argument("--ds-base-url", default="http://172.29.38.56:8765/v1", help="DeepSeek OCR base URL")
    parser.add_argument(
        "--gpt-api-key",
        default=os.environ.get("API_KEY_AZURE", ""),
        help="GPT 校正 API Key (默认读取 $API_KEY_AZURE)",
    )
    parser.add_argument(
        "--gpt-endpoint",
        default="https://ai4ae-gpt.openai.azure.com/openai/v1/",
        help="GPT 校正 endpoint",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PDF 渲染 DPI")
    parser.add_argument("--ds-model", default="deepseek-ocr-2", help="DeepSeek OCR 模型名")
    parser.add_argument("--gpt-model", default="gpt-5.2", help="GPT 校正模型名")
    parser.add_argument("--gpt-temperature", type=float, default=1.0, help="GPT 采样温度")
    parser.add_argument(
        "--no-merge-markdown",
        action="store_true",
        help="不生成合并 Markdown（默认会在工作目录根生成 merged.md，并合并所有页 result.md）",
    )
    parser.add_argument(
        "--merged-filename",
        default="merged.md",
        help="合并后的 Markdown 文件名（写入 output 目录）",
    )

    args = parser.parse_args(argv)

    from deepseek_pdf_ocr.pipeline import run_pipeline

    run_pipeline(
        pdf_path=args.pdf_path,
        ds_api_key=args.ds_api_key,
        ds_base_url=args.ds_base_url,
        gpt_api_key=args.gpt_api_key,
        gpt_endpoint=args.gpt_endpoint,
        dpi=args.dpi,
        ds_model=args.ds_model,
        gpt_model=args.gpt_model,
        gpt_temperature=args.gpt_temperature,
        merge_markdown=not args.no_merge_markdown,
        merged_filename=args.merged_filename,
    )


if __name__ == "__main__":
    main()