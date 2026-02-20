"""
deepseek-pdf-ocr: PDF OCR pipeline powered by DeepSeek OCR + GPT correction.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def run_pipeline(*args, **kwargs):
    """惰性导入，避免 import 包时强依赖重模块。"""
    from deepseek_pdf_ocr.pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


def load_ocr_results_for_llm(*args, **kwargs):
    """惰性导入，避免 import 包时强依赖重模块。"""
    from deepseek_pdf_ocr.loader import load_ocr_results_for_llm as _load

    return _load(*args, **kwargs)


__all__ = ["run_pipeline", "load_ocr_results_for_llm"]

__version__ = "0.1.0"