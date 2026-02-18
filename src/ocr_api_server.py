# %% [markdown]
# > https://github.com/deepseek-ai/DeepSeek-VL2/issues/45

# %%
# %mkdir -p local_libs
# %pip install transformers==4.47.1 "tokenizers>=0.21,<0.22" "huggingface-hub>=0.24.0,<1.0" --target=./local_libs --no-deps

# %%
import os

if not os.path.exists('./local_libs'):
    os.makedirs('local_libs', exist_ok=True)
    os.system('pip install transformers==4.47.1 "tokenizers>=0.21,<0.22" "huggingface-hub>=0.24.0,<1.0" --target=./local_libs --no-deps')

import os,sys

local_lib_path = os.path.join(os.getcwd(), 'local_libs')
if local_lib_path not in sys.path:
    sys.path.insert(0, local_lib_path)

import transformers
assert transformers.__version__ == "4.47.1"

# %%
import os

def patch_deepseek_ocr_code(model_dir="./DeepSeek-OCR-2"):
    target_file = os.path.join(model_dir, "modeling_deepseekocr2.py")
    if not os.path.exists(target_file):
        print(f"错误: 找不到文件 {target_file}")
        return
    
    # 新的锚点：在处理之前保存原始文本
    anchor = 'matches_ref, matches_images, mathes_other = re_match(outputs)'
    patch = '\n            raw_text = outputs  # [Patch] Save original text before processing\n            '
    
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 先添加 raw_text 变量
    if 'raw_text = outputs  # [Patch]' not in content:
        if anchor not in content:
            print("警告: 未找到第一个锚点")
            return
        content = content.replace(anchor, patch + anchor)
        print("添加了 raw_text 变量")
    
    # 然后修改返回值
    anchor2 = 'result.save(f"{output_path}/result_with_boxes.jpg")'
    patch2 = '\n            # [Patch] Return original text when save_results=True\n            return raw_text\n'
    
    if 'return raw_text' in content.split(anchor2)[-1][:200]:
        print("文件已完全修改")
        return
    
    if anchor2 not in content:
        print("警告: 未找到第二个锚点")
        return
    
    content = content.replace(anchor2, anchor2 + patch2)
    
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"成功修改 {target_file}")

# %%
local_dir="./DeepSeek-OCR-2"

if not os.path.exists(local_dir):
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(
        "deepseek-ai/DeepSeek-OCR-2",
        local_dir="./DeepSeek-OCR-2",
        local_dir_use_symlinks=False,
    )
    patch_deepseek_ocr_code()

# %%
# ocr_api_server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Literal
from transformers import AutoModel, AutoTokenizer
import torch
import os
import base64
from io import BytesIO
from PIL import Image
import requests
import uuid
import time

# 初始化模型
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(
    local_dir, 
    _attn_implementation='flash_attention_2', 
    trust_remote_code=True, 
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)
print("Model loaded successfully!")

# FastAPI app
app = FastAPI(title="DeepSeek-OCR-2 OpenAI Compatible API")

# 数据模型
class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: dict  # {"url": "data:image/jpeg;base64,..."}

class TextContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: str
    content: Union[str, List[Union[TextContent, ImageContent]]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

# 辅助函数
def decode_image(image_data: str) -> Image.Image:
    """从 base64 或 URL 解码图像"""
    if image_data.startswith('data:image'):
        # Base64 格式
        base64_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(base64_data)
        return Image.open(BytesIO(image_bytes))
    elif image_data.startswith('http'):
        # URL 格式
        response = requests.get(image_data)
        return Image.open(BytesIO(response.content))
    else:
        raise ValueError("Unsupported image format")

def extract_image_and_prompt(messages: List[Message]) -> tuple:
    """从消息中提取图像和提示词"""
    image = None
    prompt_text = ""
    
    for message in messages:
        if isinstance(message.content, str):
            prompt_text += message.content + "\n"
        elif isinstance(message.content, list):
            for content in message.content:
                if content.type == "text":
                    prompt_text += content.text + "\n"
                elif content.type == "image_url":
                    image_url = content.image_url.get("url", "")
                    if image_url and not image:
                        image = decode_image(image_url)
    
    # 默认提示词
    if not prompt_text.strip():
        prompt_text = "Convert the document to markdown."
    
    return image, prompt_text.strip()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # 提取图像和提示词
        image, user_prompt = extract_image_and_prompt(request.messages)
        
        if image is None:
            raise HTTPException(status_code=400, detail="No image provided in messages")
        
        # 保存临时图像
        temp_image_path = f"/tmp/ocr_temp_{uuid.uuid4()}.png"
        image.save(temp_image_path)
        
        # 构建 OCR prompt
        ocr_prompt = f"<image>\n<|grounding|>{user_prompt}"
        
        # 执行 OCR
        output_path = '/tmp/ocr_output'
        os.makedirs(output_path, exist_ok=True)
        
        result = model.infer(
            tokenizer, 
            prompt=ocr_prompt, 
            image_file=temp_image_path, 
            output_path=output_path, 
            base_size=1024, 
            image_size=768, 
            crop_mode=True, 
            save_results=True  # 不保存中间结果
        )
        
        assert result is not None
        # 清理临时文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        # 构建响应
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": len(user_prompt.split()),
                "completion_tokens": len(result.split()),
                "total_tokens": len(user_prompt.split()) + len(result.split())
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-ocr-2",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepseek"
            }
        ]
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
        import socket

        def get_local_ip():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
                return ip
            except Exception:
                return "127.0.0.1"

        PORT = 8765
        local_ip = get_local_ip()
        print(f"""
#{'='*50}
#🚀 DeepSeek-OCR-2 Server Running
#Local:  http://127.0.0.1:{PORT}/v1
#Remote: http://{local_ip}:{PORT}/v1
#{'='*50}
# #Test with:""")

        print(f"""
from openai import OpenAI
import base64

client = OpenAI(api_key="NVIDIA RTX A6000", base_url="http://{local_ip}:{PORT}/v1")

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

response = client.chat.completions.create(
    model="deepseek-ocr-2",
    messages=[{{"role": "user", "content": [
        {{"type": "text", "text": "Convert the document to markdown."}},
        {{"type": "image_url", "image_url": {{"url": f"data:image/png;base64,{{encode_image('demo.png')}}"}}}}
    ]}}]
)

print(response.choices[0].message.content)
""")

if __name__ == "__main__":
    # python ocr_api_server.py
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)