import os
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from pydantic import BaseModel

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# 优先使用GPU，完全避免CPU计算
print("🔧 正在配置GPU优化设置...")

# 设置环境变量强制使用GPU
# 设置环境变量，让Docker环境变量生效
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 注释掉硬编码，让Docker环境变量生效
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免tokenizer的CPU并行

# 极度限制CPU线程使用，强制使用GPU
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# 检查GPU可用性并打印详细信息
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    current_gpu = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu)
    gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / (1024**3)
    
    print("🎮 GPU设备信息:")
    print(f"   - 可用GPU数量: {gpu_count}")
    print(f"   - 当前使用GPU: {current_gpu}")
    print(f"   - GPU名称: {gpu_name}")
    print(f"   - GPU总显存: {gpu_memory:.2f} GB")
    
    # 启用所有GPU优化选项
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # 优化卷积操作
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    
    # 设置GPU内存策略
    torch.cuda.empty_cache()  # 清空缓存
    print("✅ GPU优化设置完成")
else:
    print("❌ 警告: 未检测到CUDA GPU，将使用CPU（性能会很差）")

app = FastAPI(title="BGE Model Service")

# 添加CORS中间件以允许跨域访问
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境中应该限制具体域名）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型路径
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/app/models/bge-large-zh-v1.5")
RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "/app/models/bge-reranker-large")
CHAT_MODEL_PATH = os.getenv("CHAT_MODEL_PATH", "/app/models/gpt-oss-20b")

# 设备常量
CUDA_DEVICE = "cuda:0"
DEFAULT_DEVICE = CUDA_DEVICE if torch.cuda.is_available() else "cpu"

# 全局变量存储模型和tokenizer
embedding_tokenizer = None
embedding_model = None
reranker_tokenizer = None
reranker_model = None
chat_tokenizer = None
chat_model = None

# LangChain 相关变量
langchain_llm = None
langchain_chat_model = None
chat_prompt_template = None

# -------------------- 监控与统计辅助函数 --------------------
def _print_gpu_status():
    """打印详细的GPU使用状态"""
    if not torch.cuda.is_available():
        print("❌ GPU不可用，使用CPU")
        return
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = props.total_memory / (1024**3)
        
        print(f"📊 GPU {i} ({props.name}):")
        print(f"   - 已分配显存: {allocated:.2f} GB")
        print(f"   - 已保留显存: {reserved:.2f} GB") 
        print(f"   - 总显存: {total:.2f} GB")
        print(f"   - 显存使用率: {(allocated/total)*100:.1f}%")

def _check_gpu_utilization():
    """检查GPU利用率（需要nvidia-ml-py库）"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu
    except Exception:
        return None

def _gpu_memory_snapshot():
    if not torch.cuda.is_available():
        return []
    snapshots = []
    num = torch.cuda.device_count()
    for idx in range(num):
        try:
            torch.cuda.synchronize(idx)
        except Exception:
            pass
        stats = {}
        stats["device_index"] = idx
        stats["name"] = torch.cuda.get_device_name(idx)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            stats["total_GB"] = round(total_bytes / 1024**3, 2)
            stats["free_GB"] = round(free_bytes / 1024**3, 2)
        except Exception:
            stats["total_GB"] = stats["free_GB"] = None
        stats["allocated_GB"] = round(torch.cuda.memory_allocated(idx) / 1024**3, 2)
        stats["reserved_GB"] = round(torch.cuda.memory_reserved(idx) / 1024**3, 2)
        
        # 尝试获取GPU利用率
        gpu_util = _check_gpu_utilization() if idx == 0 else None
        if gpu_util is not None:
            stats["gpu_utilization"] = gpu_util
            
        snapshots.append(stats)
    return snapshots

def _model_param_stats():
    def _count(m):
        if m is None:
            return None
        try:
            return sum(p.numel() for p in m.parameters())
        except Exception:
            return None
    return {
        "embedding_params": _count(embedding_model),
        "reranker_params": _count(reranker_model),
        "chat_params": _count(chat_model),
    }

def _process_memory():
    rss = None
    try:
        # 尝试 psutil
        import psutil  # type: ignore
        p = psutil.Process(os.getpid())
        rss = p.memory_info().rss
    except Exception:
        # Linux /proc/self/statm 回退
        try:
            with open("/proc/self/statm", "r") as f:
                parts = f.read().strip().split()
                if len(parts) >= 2:
                    page_size = os.sysconf("SC_PAGE_SIZE")
                    rss = int(parts[1]) * page_size
        except Exception:
            rss = None
    if rss is None:
        return None
    return {
        "rss_GB": round(rss / 1024**3, 3)
    }

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

class ChatRequest(BaseModel):
    messages: List[dict]  # [{"role": "user", "content": "..."}]
    max_length: int = 2048
    temperature: float = 0.7
    do_sample: bool = True

class ChatResponse(BaseModel):
    message: str
    model: str

def _load_embedding():
    global embedding_tokenizer, embedding_model
    print(f"📥 正在加载Embedding模型: {EMBEDDING_MODEL_PATH}")
    
    embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
    embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)
    
    # 移动到GPU（如果可用）
    if torch.cuda.is_available():
        embedding_model = embedding_model.to(DEFAULT_DEVICE)
        print(f"✅ Embedding模型已加载到: {next(embedding_model.parameters()).device}")
        print(f"📊 模型数据类型: {next(embedding_model.parameters()).dtype}")
    else:
        print("⚠️  警告: Embedding模型加载到CPU")
    
    embedding_model.eval()
    _print_gpu_status()

def _load_reranker():
    global reranker_tokenizer, reranker_model
    print(f"📥 正在加载Reranker模型: {RERANKER_MODEL_PATH}")
    
    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
    reranker_model = AutoModel.from_pretrained(RERANKER_MODEL_PATH)
    
    # 移动到GPU（如果可用）
    if torch.cuda.is_available():
        reranker_model = reranker_model.to(DEFAULT_DEVICE)
        print(f"✅ Reranker模型已加载到: {next(reranker_model.parameters()).device}")
        print(f"📊 模型数据类型: {next(reranker_model.parameters()).dtype}")
    else:
        print("⚠️  警告: Reranker模型加载到CPU")
    
    reranker_model.eval()
    _print_gpu_status()

def _load_chat():
    global chat_tokenizer, chat_model, langchain_llm, langchain_chat_model, chat_prompt_template
    print(f"📥 正在加载Chat模型: {CHAT_MODEL_PATH}")
    print("🔧 使用模型原生配置")
    
    chat_tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_PATH)
    
    # 强制GPU设备映射策略
    if torch.cuda.is_available():
        print("🎮 配置GPU设备映射...")
        
        # 检查GPU显存
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"💾 可用GPU显存: {gpu_memory_gb:.2f} GB")
        
        # 根据显存大小选择策略
        if gpu_memory_gb >= 24:  # 24GB以上显存
            device_map = {"": 0}  # 全部放在GPU 0
            print("📍 策略: 所有层放在GPU 0")
        elif gpu_memory_gb >= 12:  # 12-24GB显存
            device_map = "auto"  # 自动分配但优先GPU
            print("📍 策略: 自动分配，优先GPU")
        else:  # 显存不足
            print("⚠️  警告: GPU显存不足12GB，可能影响性能")
            device_map = "auto"
    else:
        device_map = "cpu"
        print("❌ 使用CPU设备映射")
    
    # 加载模型，使用最简配置
    print("🔄 开始调用 AutoModelForCausalLM.from_pretrained...")
    print(f"📁 模型路径: {CHAT_MODEL_PATH}")
    
    import time
    start_time = time.time()
    
    try:
        # 使用accelerate的自动设备映射
        print("⏳ 正在实例化模型...")
        chat_model = AutoModelForCausalLM.from_pretrained(
            CHAT_MODEL_PATH,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,  # 明确指定数据类型
            device_map="auto"  # 让accelerate自动管理设备映射
        )
        
        end_time = time.time()
        print(f"✅ 模型加载完成，耗时: {end_time - start_time:.2f} 秒")
        
        # 不需要手动移动到GPU，accelerate已经处理了
        print("✅ 模型已通过accelerate自动映射到GPU")
        
        # 检查显存使用
        _print_gpu_status()
        
        print("🔄 开始检查模型状态...")
        
    except Exception as e:
        print(f"❌ 模型加载过程中出错: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    
    print("🔄 模型加载完成，开始后处理...")
    chat_model.eval()
    print("✅ 模型设置为评估模式")
    
    # 先打印基本的模型加载状态
    print("✅ Chat模型已加载到设备")
    print("📊 Chat模型设备分布:")
    try:
        if hasattr(chat_model, 'hf_device_map'):
            device_map_items = list(chat_model.hf_device_map.items())[:5]  # 只显示前5个避免输出过多
            for layer, device in device_map_items:
                if isinstance(layer, str) and len(layer) > 50:
                    layer = layer[:47] + "..."
                print(f"   {layer}: {device}")
            if len(chat_model.hf_device_map) > 5:
                print(f"   ... 和其他 {len(chat_model.hf_device_map) - 5} 个层")
        else:
            print("   未找到设备映射信息")
    except Exception as e:
        print(f"   设备映射检查失败: {e}")
    
    print("🔍 开始检查参数分布...")
    # 验证模型是否在GPU上
    try:
        gpu_params = 0
        cpu_params = 0
        total_checked = 0
        
        for name, param in chat_model.named_parameters():
            if param.device.type == 'cuda':
                gpu_params += 1
            else:
                cpu_params += 1
            total_checked += 1
            
            # 每检查1000个参数打印一次进度，避免卡住
            if total_checked % 1000 == 0:
                print(f"   已检查参数: {total_checked}")
        
        total_params = gpu_params + cpu_params
        gpu_percentage = (gpu_params / total_params * 100) if total_params > 0 else 0
        
        print("🎯 模型参数分布:")
        print(f"   - GPU参数: {gpu_params}/{total_params} ({gpu_percentage:.1f}%)")
        print(f"   - CPU参数: {cpu_params}/{total_params} ({100-gpu_percentage:.1f}%)")
        
        if gpu_percentage < 80:
            print("⚠️  警告: 超过20%的参数在CPU上，这会导致性能下降！")
        else:
            print("✅ 模型主要在GPU上，性能良好")
            
    except Exception as e:
        print(f"❌ 参数分布检查失败: {e}")
    
    print("✅ Chat模型基础加载完成 (MXFP4)")
    _print_gpu_status()
    
    # 集成 LangChain
    print("🔗 正在集成 LangChain...")
    try:
        # 使用正确的 HuggingFacePipeline 初始化方式
        from transformers import pipeline

        # 创建 transformers pipeline
        hf_pipeline = pipeline(
            "text-generation",
            model=chat_model,
            tokenizer=chat_tokenizer,
            # 不指定device参数，因为模型已通过accelerate管理设备
            model_kwargs={
                "temperature": 0.3,
                "max_new_tokens": 512,
                "do_sample": True,
                "pad_token_id": chat_tokenizer.eos_token_id,
                "eos_token_id": chat_tokenizer.eos_token_id,
            }
        )
        
        # 创建 LangChain Pipeline
        langchain_llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
        # 创建 Chat 模型
        langchain_chat_model = ChatHuggingFace(llm=langchain_llm, verbose=False)
        
        # 创建对话模板
        chat_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "你是一个有用的AI助手。请简洁、准确地回答用户的问题，不要包含任何多余的内容或解释。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        print("✅ LangChain 集成完成")
        
    except Exception as e:
        print(f"⚠️ LangChain 集成失败，将使用原生模式: {e}")
        import traceback
        print("🐛 详细错误信息:")
        traceback.print_exc()
        langchain_llm = None
        langchain_chat_model = None
        chat_prompt_template = None

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
    
    print("🚀 启动BGE模型服务...")
    print("=" * 60)
    
    # 打印初始GPU状态
    print("🔍 检查GPU环境:")
    _print_gpu_status()
    
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    app.state.chat_config = {
        "quant": "mxfp4",
        "resolved_dtype": "native",
        "max_new_tokens_default": int(os.getenv("MAX_NEW_TOKENS", "512"))
    }
    
    print("🎯 配置信息:")
    print(f"   - 量化方式: {app.state.chat_config['quant']}")
    print(f"   - 数据类型: {app.state.chat_config['resolved_dtype']}")
    print(f"   - 默认最大Token: {app.state.chat_config['max_new_tokens_default']}")
    
    try:
        print("\n📦 开始加载模型...")
        _load_embedding()
        print()
        _load_reranker() 
        print()
        _load_chat()
        print()
        _maybe_warmup()
        
        print("\n✅ 所有模型加载完成！")
        print("🎮 最终GPU状态:")
        _print_gpu_status()
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 模型加载错误: {e}")
        import traceback
        print("🐛 详细错误追踪:")
        traceback.print_exc()
        print("⚠️  服务将以可用组件继续运行")
        _print_gpu_status()

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
    return {
        "gpu_memory": _gpu_memory_snapshot(),
        "models": _model_param_stats(),
        "process_memory": _process_memory(),
        "chat_config": getattr(app.state, 'chat_config', {}),
    }

@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: TextRequest):
    if embedding_model is None or embedding_tokenizer is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")
    
    try:
        # 编码文本
        encoded_input = embedding_tokenizer(
            request.texts,
            padding=True,
            truncation=True,
            max_length=request.max_length,
            return_tensors='pt'
        )
        
        # 将输入移动到模型所在的设备
        device = next(embedding_model.parameters()).device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # 获取嵌入向量
        with torch.no_grad():
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
        device = next(reranker_model.parameters()).device
        scores = []
        for doc in request.documents:
            # 为重排序模型准备输入对
            pairs = [[request.query, doc]]
            
            # 编码查询-文档对
            encoded_input = reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=request.max_length,
                return_tensors='pt'
            )
            
            # 将输入移动到模型所在的设备
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            # 获取重排序分数
            with torch.no_grad():
                outputs = reranker_model(**encoded_input)
                # 获取CLS token的输出作为相关性分数
                score = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
                # 计算相关性分数（这里使用简单的均值，实际可能需要根据具体模型调整）
                relevance_score = float(np.mean(score))
                scores.append(relevance_score)
        
        return RerankResponse(scores=scores)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing rerank request: {str(e)}")

def _get_model_device(model):
    """获取模型的设备"""
    device = None
    if torch.cuda.is_available():
        try:
            # 检查模型是否有device_map
            if hasattr(model, 'hf_device_map'):
                # 获取第一个可用设备
                device = next(iter(model.hf_device_map.values()))
                if isinstance(device, str):
                    device = torch.device(device)
                elif isinstance(device, int):
                    device = torch.device(f"cuda:{device}")
            else:
                # 尝试从模型参数获取设备
                device = next(p.device for p in model.parameters() if p.device.type != "meta")
        except (StopIteration, AttributeError):
            device = torch.device(DEFAULT_DEVICE)
    else:
        device = torch.device("cpu")
    return device

def _generate_with_model(model, tokenizer, input_ids, attention_mask, max_new_tokens, temperature, do_sample, device):
    """使用模型生成文本"""
    with torch.no_grad():
        if torch.cuda.is_available() and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    early_stopping=True
                )
        else:
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                early_stopping=True
            )
    return outputs

def _chat_with_langchain(messages: List[dict]):
    """使用 LangChain 进行聊天推理"""
    if langchain_chat_model is None or chat_prompt_template is None:
        return None
    
    try:
        # 转换消息格式
        chat_history = []
        current_input = ""
        
        for msg in messages:
            if msg["role"] == "user":
                current_input = msg["content"]  # 最后一个用户输入
            elif msg["role"] == "assistant":
                if current_input:  # 如果有之前的用户输入
                    chat_history.append(HumanMessage(content=current_input))
                    current_input = ""
                chat_history.append(AIMessage(content=msg["content"]))
        
        # 构建prompt
        formatted_prompt = chat_prompt_template.format_messages(
            chat_history=chat_history,
            input=current_input
        )
        
        # 使用 LangChain 生成回复
        response = langchain_chat_model.invoke(formatted_prompt)
        
        # 提取回复内容
        if hasattr(response, 'content'):
            return response.content.strip()
        else:
            return str(response).strip()
            
    except Exception as e:
        print(f"❌ LangChain 推理失败: {e}")
        return None

@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    if chat_model is None or chat_tokenizer is None:
        raise HTTPException(status_code=500, detail="Chat model not loaded")
    
    # 打印推理开始状态
    print(f"🚀 开始Chat推理，输入长度: {len(request.messages)} 条消息")
    _print_gpu_status()
    
    try:
        # 优先尝试使用 LangChain
        if langchain_chat_model is not None:
            print("🔗 使用 LangChain 模式推理")
            assistant_response = _chat_with_langchain(request.messages)
            
            if assistant_response:
                print("✅ LangChain 推理完成")
                _print_gpu_status()
                return ChatResponse(
                    message=assistant_response,
                    model=CHAT_MODEL_PATH.split("/")[-1] + " (LangChain)"
                )
            else:
                print("⚠️ LangChain 推理失败，回退到原生模式")
        
        # 回退到原生模式
        print("🔧 使用原生模式推理")
        # 构建对话prompt
        conversation = ""
        for message in request.messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                conversation += f"User: {content}\n"
            elif role == "assistant":
                conversation += f"Assistant: {content}\n"
        
        conversation += "Assistant: "
        print(f"📝 对话上下文长度: {len(conversation)} 字符")
        
        # 编码输入 - 强制使用GPU友好的设置
        encoded = chat_tokenizer(
            conversation,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        
        print(f"🔢 Token数量: {encoded['input_ids'].shape[1]}")
        
        # 获取设备并移动数据到GPU
        device = _get_model_device(chat_model)
        print(f"📱 推理设备: {device}")
        
        # 确保输入在正确设备上
        input_ids = encoded["input_ids"].to(device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
        
        print("✅ 输入数据已移动到GPU")
        
        # 生成回复 - 使用优化的生成函数
        max_new_tokens = min(request.max_length, getattr(app.state, 'chat_config', {}).get('max_new_tokens_default', request.max_length))
        print(f"🎯 最大生成Token数: {max_new_tokens}")
        
        # 开始推理并监控
        print("⚡ 开始GPU推理...")
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
            
        outputs = _generate_with_model(
            chat_model, chat_tokenizer, input_ids, attention_mask,
            max_new_tokens, request.temperature, request.do_sample, device
        )
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time) / 1000.0  # 转换为秒
            print(f"⏱️  GPU推理时间: {inference_time:.3f} 秒")
        
        # 解码输出
        response = chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response[len(conversation):]
        
        # 计算生成的token数
        generated_tokens = outputs[0].shape[0] - input_ids.shape[1]
        print(f"📊 生成Token数: {generated_tokens}")
        
        if start_time and generated_tokens > 0:
            tokens_per_second = generated_tokens / inference_time
            print(f"🚀 生成速度: {tokens_per_second:.2f} tokens/秒")
        
        print("✅ Chat推理完成")
        _print_gpu_status()
        
        return ChatResponse(
            message=assistant_response.strip(),
            model=CHAT_MODEL_PATH.split("/")[-1]
        )
    
    except Exception as e:
        print(f"❌ Chat推理失败: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 已清理GPU缓存")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

if __name__ == "__main__":
    print("🌟 启动BGE模型服务器")
    print("🎮 GPU检查:")
    _print_gpu_status()
    
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0",  # 确保绑定到所有网络接口，允许外部访问
        port=8000,       # Docker容器内部使用8000端口
        log_level="info",
        access_log=True
    )