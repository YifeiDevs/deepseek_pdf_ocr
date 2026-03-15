"""
ocr_api_server_vllm.py – DeepSeek-OCR-2 OpenAI-Compatible API (vLLM Backend)
=============================================================================

Drop-in replacement for ``ocr_api_server.py`` powered by **vLLM** for
high-throughput batch inference.  Supports **text + multiple images** in a
single request — each image is OCR'd independently via vLLM batching and the
results are combined in the response.

Endpoints (OpenAI-compatible)::

    POST /v1/chat/completions   – OCR one or more images
    GET  /v1/models             – list available models
    GET  /health                – liveness probe

Environment variables (all optional, defaults shown)::

    DEEPSEEK_OCR_MODEL_DIR   ./DeepSeek-OCR-2      model weights directory
    DEEPSEEK_OCR_VLLM_DIR                           vLLM adapter code path
    CUDA_VISIBLE_DEVICES     0                       GPU selection
    VLLM_MAX_NUM_SEQS        100                     max concurrent sequences
    VLLM_TP_SIZE             1                       tensor-parallel GPUs
    VLLM_GPU_UTIL            0.75                    GPU memory fraction
    OCR_PORT                 8765                    HTTP port

Usage::

    python ocr_api_server_vllm.py
"""
from __future__ import annotations

import base64
import os
import socket
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, Literal, Optional, Union

import torch
from PIL import Image

# ╔══════════════════════════════════════════════════════════════╗
# ║  1. Environment  (MUST precede any vLLM / triton import)   ║
# ╚══════════════════════════════════════════════════════════════╝
if getattr(torch.version, "cuda", None) == "11.8":
    os.environ.setdefault(
        "TRITON_PTXAS_PATH", "/usr/local/cuda-11.8/bin/ptxas"
    )
os.environ["VLLM_USE_V1"] = "0"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Add the vLLM adapter package to sys.path when supplied
_adapter_dir = os.environ.get("DEEPSEEK_OCR_VLLM_DIR", "")
if _adapter_dir and _adapter_dir not in sys.path:
    sys.path.insert(0, _adapter_dir)

sys.path.append(r"/home/shunshunliu/yifei/OCR/DS-OCR-v2/vllm/DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm")

# ╔══════════════════════════════════════════════════════════════╗
# ║  2. vLLM + DeepSeek-OCR-2 adapter imports                  ║
# ╚══════════════════════════════════════════════════════════════╝
from vllm import LLM, SamplingParams                              # noqa: E402
from vllm.model_executor.models.registry import ModelRegistry      # noqa: E402

from deepseek_ocr2 import DeepseekOCR2ForCausalLM                 # noqa: E402
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor    # noqa: E402
from process.image_process import DeepseekOCR2Processor            # noqa: E402
from config import CROP_MODE                                       # noqa: E402

# ╔══════════════════════════════════════════════════════════════╗
# ║  3. Model code patch                                        ║
# ║     Reused from ocr_api_server.patch_deepseek_ocr_code      ║
# ║     (Inlined because importing that module would trigger     ║
# ║      HF model loading, transformers version pin, etc.)       ║
# ╚══════════════════════════════════════════════════════════════╝


def patch_deepseek_ocr_code(model_dir: str = "./DeepSeek-OCR-2") -> None:
    """Patch ``modeling_deepseekocr2.py`` so that
    ``model.infer(..., save_results=True)`` returns the raw OCR text.
    Idempotent — safe to call more than once.
    """
    target = os.path.join(model_dir, "modeling_deepseekocr2.py")
    if not os.path.exists(target):
        print(f"[patch] skip — {target} not found")
        return

    with open(target, "r", encoding="utf-8") as fh:
        src = fh.read()

    # — anchor 1: save the raw text before post-processing —
    a1 = "matches_ref, matches_images, mathes_other = re_match(outputs)"
    p1 = (
        "\n            raw_text = outputs"
        "  # [Patch] Save original text before processing\n            "
    )
    if "raw_text = outputs  # [Patch]" not in src:
        if a1 not in src:
            print("[patch] anchor-1 not found")
            return
        src = src.replace(a1, p1 + a1)

    # — anchor 2: return raw text after saving the debug image —
    a2 = 'result.save(f"{output_path}/result_with_boxes.jpg")'
    p2 = (
        "\n            # [Patch] Return original text"
        " when save_results=True\n            return raw_text\n"
    )
    if "return raw_text" in src.split(a2)[-1][:200]:
        print("[patch] already applied")
        return
    if a2 not in src:
        print("[patch] anchor-2 not found")
        return
    src = src.replace(a2, a2 + p2)

    with open(target, "w", encoding="utf-8") as fh:
        fh.write(src)
    print(f"[patch] ✓ patched {target}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  4. Download model weights (first run) & apply patch        ║
# ╚══════════════════════════════════════════════════════════════╝
MODEL_DIR: str = os.environ.get("DEEPSEEK_OCR_MODEL_DIR", "./DeepSeek-OCR-2")

if not os.path.exists(MODEL_DIR):
    from huggingface_hub import snapshot_download  # noqa: E402

    MODEL_DIR = snapshot_download(
        "deepseek-ai/DeepSeek-OCR-2",
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )
    patch_deepseek_ocr_code(MODEL_DIR)

# ╔══════════════════════════════════════════════════════════════╗
# ║  5. Register architecture & start vLLM engine               ║
# ╚══════════════════════════════════════════════════════════════╝
ModelRegistry.register_model(
    "DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM
)

print("[vllm] Loading engine …")
llm = LLM(
    model=MODEL_DIR,
    hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
    block_size=256,
    enforce_eager=False,
    trust_remote_code=True,
    max_model_len=8192,
    swap_space=0,
    max_num_seqs=int(os.environ.get("VLLM_MAX_NUM_SEQS", "100")),
    tensor_parallel_size=int(os.environ.get("VLLM_TP_SIZE", "1")),
    gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.75")),
)
print("[vllm] ✓ Engine ready")

# Shared stateless logits processors (safe to reuse across requests)
_LOGITS_PROCS = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=20,
        window_size=90,
        whitelist_token_ids={128821, 128822},
    )
]

# ╔══════════════════════════════════════════════════════════════╗
# ║  6. FastAPI application                                      ║
# ╚══════════════════════════════════════════════════════════════╝
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="DeepSeek-OCR-2 (vLLM)")

# ── Request / Response schemas (OpenAI-compatible) ────────────


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: dict  # {"url": "data:image/…;base64,…" | "https://…"}


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: str
    content: Union[str, List[Union[TextContent, ImageContent]]]


class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-ocr-2"
    messages: List[Message]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 8192
    stream: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict


# ── Helpers ───────────────────────────────────────────────────


def _decode_image(src: str) -> Image.Image:
    """Base64 data-URI **or** HTTP(S) URL → PIL RGB Image."""
    if src.startswith("data:image"):
        raw = base64.b64decode(src.split(",", 1)[1])
        return Image.open(BytesIO(raw)).convert("RGB")
    if src.startswith(("http://", "https://")):
        import requests as _rq  # lazy — not needed for base64 only

        r = _rq.get(src, timeout=60)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    raise ValueError(f"Unsupported image source: {src[:120]}")


def _extract_content(
    messages: List[Message],
) -> tuple[str, list[Image.Image]]:
    """Parse OpenAI-style messages → ``(prompt_text, [PIL images])``."""
    texts: list[str] = []
    images: list[Image.Image] = []
    for msg in messages:
        if isinstance(msg.content, str):
            texts.append(msg.content)
        elif isinstance(msg.content, list):
            for part in msg.content:
                if part.type == "text":
                    texts.append(part.text)
                elif part.type == "image_url":
                    url = part.image_url.get("url", "")
                    if url:
                        images.append(_decode_image(url))
    prompt = "\n".join(t for t in texts if t.strip()).strip()
    return (prompt or "Convert the document to markdown."), images


def _preprocess_one(image: Image.Image, prompt: str) -> dict:
    """Tokenise **one** image on the CPU and return a vLLM input dict."""
    data = DeepseekOCR2Processor().tokenize_with_images(
        images=[image],
        bos=True,
        eos=True,
        cropping=CROP_MODE,
    )
    return {"prompt": prompt, "multi_modal_data": {"image": data}}


# ── Endpoints ─────────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OCR one **or more** images.

    Multiple ``image_url`` items are processed as an efficient vLLM
    **batch** and the results are merged into a single response.
    """
    if request.stream:
        raise HTTPException(400, "Streaming is not supported.")

    try:
        user_prompt, images = _extract_content(request.messages)
        if not images:
            raise HTTPException(400, "No image provided in the request.")

        ocr_prompt = f"<image>\n<|grounding|>{user_prompt}"

        sampling = SamplingParams(
            temperature=request.temperature or 0.0,
            max_tokens=request.max_tokens or 8192,
            logits_processors=_LOGITS_PROCS,
            skip_special_tokens=False,
        )

        # ── Parallel CPU preprocessing ──
        with ThreadPoolExecutor(max_workers=min(len(images), 16)) as pool:
            vllm_inputs: list[dict] = list(
                pool.map(
                    lambda img: _preprocess_one(img, ocr_prompt), images
                )
            )

        # ── Batch GPU inference ──
        outputs = llm.generate(vllm_inputs, sampling, use_tqdm=True)

        # ── Assemble response text ──
        if len(outputs) == 1:
            result_text = outputs[0].outputs[0].text
        else:
            parts = [
                f"<!-- image {i} -->\n{o.outputs[0].text}"
                for i, o in enumerate(outputs, 1)
            ]
            result_text = "\n\n---\n\n".join(parts)

        # Token accounting (exact, from vLLM internals)
        prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        completion_tokens = sum(
            len(o.outputs[0].token_ids) for o in outputs
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-ocr-2",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepseek",
            }
        ],
    }


@app.get("/health")
async def health_check():
    """Liveness probe."""
    return {
        "status": "ok",
        "backend": "vllm",
        "model_loaded": llm is not None,
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║  7. CLI entry-point                                          ║
# ╚══════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    import uvicorn

    def _local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    PORT = int(os.environ.get("OCR_PORT", "8765"))
    ip = _local_ip()

    print(
        f"""{'=' * 55}
🚀  DeepSeek-OCR-2  ·  vLLM Server
    Local:   http://127.0.0.1:{PORT}/v1
    Remote:  http://{ip}:{PORT}/v1
{'=' * 55}"""
    )

    # ── Print quick-test snippets for the user ──
    print(
        f"""# ── Quick test (single image) ──
from openai import OpenAI
import base64

client = OpenAI(api_key="none", base_url="http://{ip}:{PORT}/v1")

def enc(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ── Quick test ──
resp = client.chat.completions.create(
    model="deepseek-ocr-2",
    messages=[{{"role": "user", "content": [
        {{"type": "text", "text": "Convert the documents to markdown."}},
        {{"type": "image_url", "image_url": {{"url": f"data:image/png;base64,{{enc('p1.png')}}"}} }},
        {{"type": "image_url", "image_url": {{"url": f"data:image/png;base64,{{enc('p2.png')}}"}} }},
    ]}}],
)
print(resp.choices[0].message.content)"""
    )
    print(r'''# 返回格式精简说明：
# 1. 多图分割符：使用 来区分不同图片的识别结果（N为图片序号）。
# 2. 元素结构：<|ref|>元素类型<|/ref|><|det|>[[左上X, 左上Y, 右下X, 右下Y]]<|/det|> \n 具体内容。
# 3. 常见元素类型包括：image, figure_title, sub_title, text, equation, table 等。

<!-- image 1 -->
<|ref|>image<|/ref|><|det|>[[272, 20, 752, 232]]<|/det|>

<|ref|>figure_title<|/ref|><|det|>[[170, 237, 822, 267]]<|/det|>
Figure 2: (left) Scaled Dot-Product Attention...

<|ref|>text<|/ref|><|det|>[[170, 358, 823, 414]]<|/det|>
We call our particular attention "Scaled Dot-Product Attention"...

<|ref|>equation<|/ref|><|det|>[[352, 478, 821, 512]]<|/det|>
 \[  Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V \quad (1) \] 

<|ref|>table<|/ref|><|det|>[[192, 797, 802, 890]]<|/det|>
<table><tr><td>Layer Type</td><td>...</tr></table>

---

<!-- image 2 -->

<|ref|>text<|/ref|><|det|>[[272, 20, 752, 232]]<|/det|>

---

<!-- image 3 -->

<|ref|>text<|/ref|><|det|>[[...]]<|/det|>

---

<!-- image 4 -->

<|ref|>...<|/ref|><|det|>[[...],[...]]<|/det|>
...
''')
    uvicorn.run(app, host="0.0.0.0", port=PORT)
# python /home/shunshunliu/yifei/OCR/DS-OCR-v2/ocr_api_server_vllm.py