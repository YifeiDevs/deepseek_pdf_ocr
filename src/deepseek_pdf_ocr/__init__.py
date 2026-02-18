"""
deepseek-pdf-ocr: PDF OCR pipeline powered by DeepSeek OCR + GPT correction.
"""

from deepseek_pdf_ocr.pipeline import run_pipeline
from deepseek_pdf_ocr.loader import load_ocr_results_for_llm

__all__ = [
    "run_pipeline",
    "load_ocr_results_for_llm",
]

__version__ = "0.1.0"