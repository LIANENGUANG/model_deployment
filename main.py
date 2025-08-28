import os
from asyncio import Semaphore
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# GPU优化设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

print(f"🚀 GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")

app = FastAPI(title="BGE Model Service")

# 模型路径
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/app/models/bge-large-zh-v1.5")
RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "/app/models/bge-reranker-large")
CHAT_MODEL_PATH = os.getenv("CHAT_MODEL_PATH", "/app/models/qwen3-30b-a3b-instruct-2507")

# 设备常量 - H100专用
CUDA_DEVICE = "cuda:0"

# 全局变量存储模型和tokenizer
embedding_tokenizer = None
embedding_model = None
reranker_tokenizer = None
reranker_model = None
chat_tokenizer = None
chat_model = None

# KV缓存存储 - 简单dict即可
kv_cache = {}

# 性能优化常量
QWEN_TEMPLATE_CONSTANTS = {
    "IM_END": "<|im_end|>",
    "IM_START_ASSISTANT": "<|im_start|>assistant",
    "IM_START_USER": "<|im_start|>user",
    "IM_START_SYSTEM": "<|im_start|>system"
}

# API常量
CHAT_COMPLETION_CHUNK = "chat.completion.chunk"

# -------------------- 监控与统计辅助函数 --------------------
def _get_attention_config():
    """配置 Flash Attention 2"""
    try:
        print("🔍 正在检测 Flash Attention...")
        import flash_attn
        flash_version = flash_attn.__version__
        print(f"📦 检测到 Flash Attention 版本: {flash_version}")
        print(f"✅ 使用 Flash Attention {flash_version}")
        return {"attn_implementation": "flash_attention_2"}
    except ImportError:
        print("❌ Flash Attention 未安装，使用默认实现")
        return {}
    except Exception as e:
        print(f"⚠️  Flash Attention 检查失败: {e}，使用默认实现")
        return {}




class TextRequest(BaseModel):
    texts: List[str]
    max_length: int = 512

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    max_length: int = 512

class RerankResponse(BaseModel):
    scores: List[float]


# OpenAI 兼容的请求响应模型
class OpenAIMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str

class OpenAIChatRequest(BaseModel):
    model: str = "qwen3-30b-a3b-instruct-2507"
    messages: List[OpenAIMessage]
    max_tokens: int = 2048
    stream: bool = False

class OpenAIChoice(BaseModel):
    message: OpenAIMessage
    finish_reason: str = "stop"
    index: int = 0

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage


def _load_embedding():
    global embedding_tokenizer, embedding_model
    embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
    embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH).to(CUDA_DEVICE).eval()
    print("✅ Embedding模型加载完成")

def _load_reranker():
    global reranker_tokenizer, reranker_model
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Some weights of XLMRobertaModel were not initialized")
        reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
        reranker_model = AutoModel.from_pretrained(RERANKER_MODEL_PATH).to(CUDA_DEVICE).eval()
    print("✅ Reranker模型加载完成")

def _load_chat():
    global chat_tokenizer, chat_model
    chat_tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_PATH)
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    torch.cuda.empty_cache()
    
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    chat_model = AutoModelForCausalLM.from_pretrained(
        CHAT_MODEL_PATH,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
        max_memory={0: f"{int(gpu_memory_gb * 0.85)}GB"},
        **(_get_attention_config())
    ).eval()
    
    print("✅ Chat模型加载完成 (NF4量化)")

def _maybe_warmup():
    if os.getenv("CHAT_WARMUP", "0") != "1":
        return
    if chat_model is None:
        return
    try:
        print("Warmup start ...")
        t = chat_tokenizer("Hello", return_tensors="pt")
        # 找到 chat_model 的一个已加载设备
        try:
            device = next(p.device for p in chat_model.parameters() if p.device.type != "meta")
            t = {k: v.to(device) for k, v in t.items()}
        except StopIteration:
            # 如果所有参数都在 meta 设备，使用默认设备
            t = {k: v.to(DEFAULT_DEVICE) for k, v in t.items()}
        with torch.no_grad():
            chat_model.generate(**t, max_new_tokens=4)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("Warmup done")
    except Exception as e:
        print(f"Warmup failed: {e}")

@app.on_event("startup")
async def load_models():
    global embedding_tokenizer, embedding_model, reranker_tokenizer, reranker_model, chat_tokenizer, chat_model
    
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    app.state.chat_config = {
        "quant": "nf4",
        "resolved_dtype": "float16",
        "max_new_tokens_default": int(os.getenv("MAX_NEW_TOKENS", "512"))
    }
    
    try:
        print("📦 模型加载中...")
        _load_embedding()
        _load_reranker() 
        _load_chat()
        _maybe_warmup()
        
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🎮 显存使用: {allocated:.1f}GB/{total:.1f}GB ({allocated/total*100:.1f}%)")
        
    except Exception as e:
        raise e  # 直接抛出异常，不要继续运行

@app.get("/")
async def root():
    return {"message": "BGE Model Service is running"}

@app.get("/health")
async def health_check():
    # 收集GPU信息
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
            "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
            "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        }
    else:
        gpu_info = {"gpu_available": False}
    
    # 检查模型设备位置
    model_devices = {}
    if embedding_model is not None:
        model_devices["embedding_device"] = str(next(embedding_model.parameters()).device)
    if reranker_model is not None:
        model_devices["reranker_device"] = str(next(reranker_model.parameters()).device)
    if chat_model is not None:
        # 检查chat模型的设备分布
        gpu_layers = 0
        cpu_layers = 0
        for param in chat_model.parameters():
            if param.device.type == 'cuda':
                gpu_layers += 1
            else:
                cpu_layers += 1
        
        model_devices["chat_gpu_params"] = gpu_layers
        model_devices["chat_cpu_params"] = cpu_layers
        model_devices["chat_gpu_percentage"] = round(gpu_layers / (gpu_layers + cpu_layers) * 100, 1) if (gpu_layers + cpu_layers) > 0 else 0
    
    return {
        "status": "healthy", 
        "embedding_model_loaded": embedding_model is not None,
        "reranker_model_loaded": reranker_model is not None,
        "chat_model_loaded": chat_model is not None,
        "chat_quant": "mxfp4",
        "chat_dtype": getattr(app.state, 'chat_config', {}).get('resolved_dtype'),
        "max_new_tokens_default": getattr(app.state, 'chat_config', {}).get('max_new_tokens_default'),
        "gpu_info": gpu_info,
        "model_devices": model_devices
    }

@app.get("/stats")
async def stats():
    """简化的统计信息"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
            "reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        }
    
    return {
        "gpu_memory": gpu_info,
        "models_loaded": {
            "embedding": embedding_model is not None,
            "reranker": reranker_model is not None,
            "chat": chat_model is not None
        },
        "chat_config": getattr(app.state, 'chat_config', {}),
    }

@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: TextRequest):
    if embedding_model is None or embedding_tokenizer is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")
    
    try:
        # 直接处理 - 专注单请求性能
        encoded_input = embedding_tokenizer(
            request.texts,
            padding=True,
            truncation=True,
            max_length=request.max_length,
            return_tensors='pt'
        )
        
        # 移动到GPU
        encoded_input = {k: v.to(CUDA_DEVICE, non_blocking=True) for k, v in encoded_input.items()}
        
        # GPU推理
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                model_output = embedding_model(**encoded_input)
            embeddings = model_output.last_hidden_state[:, 0].cpu().numpy()
        
        return EmbeddingResponse(embeddings=embeddings.tolist())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    if reranker_model is None or reranker_tokenizer is None:
        raise HTTPException(status_code=500, detail="Reranker model not loaded")
    
    try:
        scores = []
        
        # 批量处理所有文档对 - 单请求优化
        all_pairs = [[request.query, doc] for doc in request.documents]
        
        # 一次性编码所有查询-文档对
        encoded_input = reranker_tokenizer(
            all_pairs,
            padding=True,
            truncation=True,
            max_length=request.max_length,
            return_tensors='pt'
        )
        
        # 移动到GPU
        encoded_input = {k: v.to(CUDA_DEVICE, non_blocking=True) for k, v in encoded_input.items()}
        
        # GPU推理
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = reranker_model(**encoded_input)
            
            # 批量计算相关性分数
            cls_outputs = outputs.last_hidden_state[:, 0].cpu().numpy()
            scores = [float(np.mean(score)) for score in cls_outputs]
        
        return RerankResponse(scores=scores)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing rerank request: {str(e)}")


def _generate_with_model(model, tokenizer, input_ids, attention_mask, max_new_tokens, past_key_values=None):
    """使用模型生成文本 - 业务场景使用贪婪解码确保确定性结果"""
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 贪婪解码，确定性结果
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                output_attentions=False,
                output_hidden_states=False,
                return_dict_in_generate=True
            )
    return outputs

def _stream_generate_with_model(model, tokenizer, input_ids, attention_mask, max_new_tokens, past_key_values=None):
    """优化的流式生成文本 - 业务场景确定性结果"""
    import json
    import threading
    import time
    import uuid

    from transformers import TextIteratorStreamer

    # 创建响应ID
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_time = int(time.time())
    
    try:
        # 使用TextIteratorStreamer进行流式生成
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # 业务场景生成参数 - 确定性结果
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,  # 贪婪解码，确定性结果
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "streamer": streamer,
            "num_beams": 1,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
        }
        
        # 在单独线程中启动生成
        def generate():
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    model.generate(**generation_kwargs)
        
        generation_thread = threading.Thread(target=generate)
        generation_thread.start()
        
        # 批量文本发送优化
        text_buffer = []
        for new_text in streamer:
            if new_text:
                text_buffer.append(new_text)
                if len(text_buffer) >= 3:  # 每3个token批量发送
                    chunk = {
                        "id": response_id,
                        "object": CHAT_COMPLETION_CHUNK,
                        "created": created_time,
                        "model": "qwen3-30b-a3b-instruct-2507",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": ''.join(text_buffer)},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    text_buffer = []
        
        # 发送剩余文本
        if text_buffer:
            chunk = {
                "id": response_id,
                "object": CHAT_COMPLETION_CHUNK,
                "created": created_time,
                "model": "qwen3-30b-a3b-instruct-2507",
                "choices": [{
                    "index": 0,
                    "delta": {"content": ''.join(text_buffer)},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # 等待生成线程完成
        generation_thread.join()
        
        # 发送结束chunk
        final_chunk = {
            "id": response_id,
            "object": CHAT_COMPLETION_CHUNK,
            "created": created_time,
            "model": "qwen3-30b-a3b-instruct-2507",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        
    except Exception as e:
        # 发送错误信息
        error_chunk = {
            "id": response_id,
            "object": CHAT_COMPLETION_CHUNK,
            "created": created_time,
            "model": "qwen3-30b-a3b-instruct-2507",
            "choices": [{
                "index": 0,
                "delta": {"content": f"[ERROR: {str(e)}]"},
                "finish_reason": "error"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    
    # 发送结束标记
    yield "data: [DONE]\n\n"



@app.post("/v1/chat/completions")
async def openai_chat_completion(request: OpenAIChatRequest):
    """OpenAI兼容的聊天完成API - 支持流式和非流式"""
    if chat_model is None or chat_tokenizer is None:
        raise HTTPException(status_code=500, detail="Chat model not loaded")
    
    try:
        # 转换OpenAI格式到内部格式
        internal_messages = []
        for msg in request.messages:
            internal_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # KV缓存优化
        conversation = ""
        cache_key = None
        past_key_values = None
        
        if len(internal_messages) > 1:
            history_text = str(internal_messages[:-1])
            cache_key = hash(history_text) % 1000000
            past_key_values = kv_cache.get(cache_key)
        
        for message in internal_messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                conversation += f"{QWEN_TEMPLATE_CONSTANTS['IM_START_SYSTEM']}\n{content}{QWEN_TEMPLATE_CONSTANTS['IM_END']}\n"
            elif role == "user":
                conversation += f"{QWEN_TEMPLATE_CONSTANTS['IM_START_USER']}\n{content}{QWEN_TEMPLATE_CONSTANTS['IM_END']}\n"
            elif role == "assistant":
                conversation += f"{QWEN_TEMPLATE_CONSTANTS['IM_START_ASSISTANT']}\n{content}{QWEN_TEMPLATE_CONSTANTS['IM_END']}\n"
        
        # 添加assistant开始标记
        conversation += f"{QWEN_TEMPLATE_CONSTANTS['IM_START_ASSISTANT']}\n"
        
        # 编码输入
        encoded = chat_tokenizer(
            conversation,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        
        # 移动数据到GPU
        input_ids = encoded["input_ids"].to(CUDA_DEVICE, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(CUDA_DEVICE, non_blocking=True)
        
        # 根据stream参数选择流式或非流式
        if request.stream:
            # 流式响应
            def generate_stream():
                try:
                    for chunk in _stream_generate_with_model(
                        chat_model, chat_tokenizer, input_ids, attention_mask,
                        request.max_tokens, past_key_values
                    ):
                        yield chunk
                except Exception as e:
                    yield f"data: {{'error': '{str(e)}'}}\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        else:
            # 非流式响应
            outputs = _generate_with_model(
                chat_model, chat_tokenizer, input_ids, attention_mask,
                request.max_tokens, past_key_values
            )
            
            # 解码输出
            response = chat_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            assistant_response = response[len(conversation):]
            
            # 更新KV缓存
            if cache_key is not None and hasattr(outputs, 'past_key_values') and len(kv_cache) < 20:
                kv_cache[cache_key] = outputs.past_key_values
            
            # 清理Qwen3特殊标记
            assistant_response = assistant_response.replace("<|im_end|>", "").strip()
            assistant_response = assistant_response.replace("<|im_start|>assistant", "").strip()
            
            # 计算token数量
            input_tokens = len(chat_tokenizer.encode(conversation))
            output_tokens = len(chat_tokenizer.encode(assistant_response))
            
            # 生成响应ID和时间戳
            import time
            import uuid
            
            response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created_time = int(time.time())
            
            # 返回OpenAI格式响应
            return OpenAIChatResponse(
                id=response_id,
                created=created_time,
                model=request.model,
                choices=[
                    OpenAIChoice(
                        message=OpenAIMessage(
                            role="assistant",
                            content=assistant_response
                        ),
                        finish_reason="stop",
                        index=0
                    )
                ],
                usage=OpenAIUsage(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens
                )
            )
        
    except Exception as e:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"Error processing OpenAI chat request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")