from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import sys
from pathlib import Path

# 禁用 AMD GPU 在 WSL 下的 Flash Attention 相关优化，避免崩溃
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "0"
# 禁用 SDP (Scaled Dot Product) attention 优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# 禁用某些可能导致 WSL + AMD GPU 崩溃的优化
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

# 导入模型相关库（根据您的模型类型调整）
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    print("成功导入 transformers 和 torch")
except Exception as e:
    print(f"警告: 导入 transformers 或 torch 时出错: {str(e)}")
    print(f"错误类型: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    print("请检查安装: pip install transformers torch")
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None

app = FastAPI(title="医疗模型API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储模型和tokenizer
model = None
tokenizer = None
model_path = None

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str

class AssessRequest(BaseModel):
    history: List[Dict[str, str]] = []

class AssessResponse(BaseModel):
    assessment: str

def load_model(path: str):
    """加载模型"""
    global model, tokenizer, model_path
    
    # 检查必要的库是否已导入
    if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
        raise ImportError(
            "transformers 或 torch 未正确安装。请运行: pip install transformers torch"
        )
    
    if not os.path.exists(path):
        raise ValueError(f"模型路径不存在: {path}")
    
    print(f"正在加载模型: {path}")
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        
        # 设置pad_token（如果不存在）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            attn_implementation="eager"  # 禁用 Flash Attention，使用传统 attention
        )
        
        if not torch.cuda.is_available():
            model = model.to("cpu")
        
        # 对于 AMD GPU，尝试禁用可能导致问题的 attention 优化
        if torch.cuda.is_available() and hasattr(model, 'config'):
            # 确保使用 eager attention
            if hasattr(model.config, 'attn_implementation'):
                model.config.attn_implementation = "eager"
            # 禁用 Flash Attention
            if hasattr(model.config, '_attn_implementation'):
                model.config._attn_implementation = "eager"
        
        model.eval()
        model_path = path
        print("模型加载成功！")
        print(f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def generate_response(user_message: str, history: List[Dict[str, str]] = None) -> str:
    """生成回复 - 支持Qwen3对话格式"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise ValueError("模型未加载，请先配置模型路径")
    
    try:
        # 构建对话历史
        if history is None:
            history = []
        
        # 准备消息列表（使用最近10轮对话以保持上下文）
        messages = []
        
        # 添加系统提示（可选）
        # messages.append({"role": "system", "content": "你是一个专业的医疗助手，专注于内科领域。"})
        
        # 添加历史对话（只使用最近10轮）
        for msg in history[-10:]:
            # 转换角色名称：user -> user, assistant -> assistant
            role = msg["role"]
            if role == "assistant":
                role = "assistant"
            elif role == "user":
                role = "user"
            messages.append({"role": role, "content": msg["content"]})
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        # 使用tokenizer的apply_chat_template方法格式化对话（Qwen3格式）
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            # 使用chat template格式化
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # 编码输入
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        else:
            # 如果没有chat template，使用简单格式
            conversation = ""
            for msg in messages:
                if msg["role"] == "user":
                    conversation += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                elif msg["role"] == "assistant":
                    conversation += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            conversation += "<|im_start|>assistant\n"
            inputs = tokenizer(conversation, return_tensors="pt", add_special_tokens=False)
        
        # 如果使用CPU，需要确保tensor在CPU上
        if not torch.cuda.is_available():
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
        else:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        # 生成回复
        # 对于 AMD GPU + WSL，禁用某些可能导致崩溃的优化
        # 使用 inference_mode 而不是 no_grad，在某些情况下更稳定
        with torch.inference_mode():
            # 尝试禁用 SDP attention 优化（如果可用）
            try:
                if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'sdp_kernel'):
                    torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
            except Exception:
                pass  # 如果设置失败，继续使用默认设置
            
            # 在生成前同步 CUDA 操作（对于 AMD GPU + WSL 可能有助于稳定性）
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            
            generate_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": 512,  # 增加生成长度以适应医疗诊断
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.8,
                "repetition_penalty": 1.05,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,  # 保持缓存以提高性能
            }
            
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask
            
            # 设置pad_token_id
            if tokenizer.pad_token_id is not None:
                generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
            else:
                generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
            
            try:
                outputs = model.generate(**generate_kwargs)
            except RuntimeError as e:
                # 如果生成失败，尝试使用更保守的设置
                if "cuda" in str(e).lower() or "hip" in str(e).lower():
                    print(f"警告: GPU 生成失败，尝试使用更保守的设置: {str(e)}")
                    # 尝试禁用缓存
                    generate_kwargs["use_cache"] = False
                    outputs = model.generate(**generate_kwargs)
                else:
                    raise
        
        # 解码回复（只取新生成的部分）
        generated_ids = outputs[0][input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 清理回复中的特殊标记
        response = response.replace("<|im_end|>", "")
        response = response.replace("<|im_start|>", "")
        response = response.replace("<|endoftext|>", "")
        # 清理 `<think></think>` 标签及其内容（使用正则表达式）
        import re
        # 匹配 `<think>`...`</think>` 或 `<think>`...`</think>`` 等格式
        response = re.sub(r'`<think>`.*?`</think>`', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'`<think>.*?</think>`', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'`<think>`.*?`</think>`', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'`<think>.*?</think>`', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # 清理多余的空格和换行
        response = response.strip()
        # 如果输出以某些标记开头，也去掉
        if response.startswith("assistant\n"):
            response = response.replace("assistant\n", "", 1)
        response = response.strip()
        
        return response
        
    except Exception as e:
        print(f"生成回复时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def generate_assessment(history: List[Dict[str, str]] = None) -> str:
    """根据历史对话生成健康评估报告"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise ValueError("模型未加载，请先配置模型路径")
    
    if history is None or len(history) == 0:
        raise ValueError("历史对话为空，无法生成评估")
    
    try:
        # 构建评估提示
        assessment_prompt = """请根据以下对话历史，对用户的身体健康状况进行全面评估，并提供以下内容：

1. **身体状况评估**：根据用户描述的症状和问题，分析可能的健康问题
2. **用药建议**：如有需要，推荐合适的药物（请注明：仅供参考，具体用药需咨询医生）
3. **保养建议**：提供日常保养、饮食、运动等方面的建议
4. **注意事项**：提醒用户需要注意的事项和何时应该就医

请以专业、清晰、易懂的方式呈现评估报告。

对话历史：
"""
        
        # 添加历史对话
        for msg in history:
            role_name = "用户" if msg["role"] == "user" else "助手"
            assessment_prompt += f"\n{role_name}：{msg['content']}\n"
        
        assessment_prompt += "\n\n请生成详细的健康评估报告："
        
        # 使用模型生成评估
        messages = [
            {"role": "user", "content": assessment_prompt}
        ]
        
        # 使用tokenizer的apply_chat_template方法格式化对话
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        else:
            conversation = f"<|im_start|>user\n{assessment_prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(conversation, return_tensors="pt", add_special_tokens=False)
        
        # 移动到正确的设备
        if not torch.cuda.is_available():
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
        else:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        # 生成评估报告
        with torch.inference_mode():
            # 尝试禁用 SDP attention 优化（如果可用）
            try:
                if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'sdp_kernel'):
                    torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
            except Exception:
                pass
            
            # 在生成前同步 CUDA 操作
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            
            generate_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": 1024,  # 评估报告需要更长的输出
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.8,
                "repetition_penalty": 1.05,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
            }
            
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask
            
            if tokenizer.pad_token_id is not None:
                generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
            else:
                generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
            
            try:
                outputs = model.generate(**generate_kwargs)
            except RuntimeError as e:
                if "cuda" in str(e).lower() or "hip" in str(e).lower():
                    print(f"警告: GPU 生成失败，尝试使用更保守的设置: {str(e)}")
                    generate_kwargs["use_cache"] = False
                    outputs = model.generate(**generate_kwargs)
                else:
                    raise
        
        # 解码回复
        generated_ids = outputs[0][input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 清理回复中的特殊标记
        response = response.replace("<|im_end|>", "")
        response = response.replace("<|im_start|>", "")
        response = response.replace("<|endoftext|>", "")
        
        # 清理 `<think>` 标签
        import re
        response = re.sub(r'`<think>`.*?`</think>`', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'`<think>.*?</think>`', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        response = response.strip()
        if response.startswith("assistant\n"):
            response = response.replace("assistant\n", "", 1)
        response = response.strip()
        
        return response
        
    except Exception as e:
        print(f"生成评估报告时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    config_path = Path(__file__).parent / "config.txt"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            path = f.read().strip()
            if path:
                try:
                    load_model(path)
                except Exception as e:
                    print(f"启动时加载模型失败: {str(e)}")
                    print("您可以通过 /load_model API 手动加载模型")

@app.get("/")
async def root():
    return {
        "message": "医疗模型API服务",
        "status": "running",
        "model_loaded": model is not None
    }

@app.post("/api/load_model")
async def load_model_endpoint(path: str):
    """加载模型API"""
    try:
        load_model(path)
        # 保存路径到配置文件
        config_path = Path(__file__).parent / "config.txt"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(path)
        return {"message": "模型加载成功", "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天API"""
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="模型未加载，请先通过 /api/load_model API 加载模型"
        )
    
    try:
        response = generate_response(request.message, request.history)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成回复失败: {str(e)}")

@app.post("/api/assess", response_model=AssessResponse)
async def assess(request: AssessRequest):
    """健康评估API - 根据历史对话生成健康评估报告"""
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="模型未加载，请先通过 /api/load_model API 加载模型"
        )
    
    if not request.history or len(request.history) == 0:
        raise HTTPException(
            status_code=400,
            detail="历史对话为空，无法生成评估"
        )
    
    try:
        assessment = generate_assessment(request.history)
        return AssessResponse(assessment=assessment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成评估报告失败: {str(e)}")

@app.get("/api/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": model_path
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

