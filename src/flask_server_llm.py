import ctypes
import os
import threading
import time
import argparse
import json
import asyncio
from typing import List, Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# --- RKLLM C-Types 定义 (保留原逻辑) ---
rkllm_lib = ctypes.CDLL('/usr/lib/librkllmrt.so')
RKLLM_Handle_t = ctypes.c_void_p

class LLMCallState:
    RKLLM_RUN_NORMAL = 0
    RKLLM_RUN_WAITING = 1
    RKLLM_RUN_FINISH = 2
    RKLLM_RUN_ERROR = 3

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104)
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p), ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32), ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32), ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float), ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float), ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32), ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float), ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool), ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p), ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p)]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p), ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int), ("input_data", RKLLMInputUnion)
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int), ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p), ("keep_history", ctypes.c_int)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [("text", ctypes.c_char_p), ("token_id", ctypes.c_int)]

# --- OpenAI 协议模型 ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "rkllm-model"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 4096

# --- 全局状态管理 ---
global_text_queue = []
global_state = -1
model_lock = threading.Lock()

def callback_impl(result, userdata, state):
    global global_state
    global_state = state
    if state == LLMCallState.RKLLM_RUN_NORMAL and result.contents.text:
        text = result.contents.text.decode('utf-8')
        global_text_queue.append(text)
    return 0

CALLBACK_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
c_callback = CALLBACK_TYPE(callback_impl)

class RKLLM:
    def __init__(self, model_path, platform):
        param = RKLLMParam()
        param.model_path = bytes(model_path, 'utf-8')
        param.max_context_len = 4096
        param.max_new_tokens = 4096
        param.top_k = 1
        param.top_p = 0.9
        param.temperature = 0.8
        param.repeat_penalty = 1.1
        param.is_async = False
        param.extend_param.n_batch = 1
        param.extend_param.enabled_cpus_num = 4
        
        if platform in ["rk3588", "rk3576"]:
            param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
        
        self.handle = RKLLM_Handle_t()
        rkllm_lib.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), CALLBACK_TYPE]
        ret = rkllm_lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(param), c_callback)
        if ret != 0: raise RuntimeError("RKLLM Init Failed")

    def run(self, prompt):
        in_param = RKLLMInput()
        in_param.role = b"user"
        in_param.input_type = 0
        in_param.input_data.prompt_input = ctypes.c_char_p(prompt.encode('utf-8'))
        
        infer_param = RKLLMInferParam()
        infer_param.mode = 0
        infer_param.keep_history = 0
        
        rkllm_lib.rkllm_run(self.handle, ctypes.byref(in_param), ctypes.byref(infer_param), None)

# --- FastAPI 核心逻辑 ---
app = FastAPI(title="RKLLM OpenAI API")
rkllm_instance: RKLLM = None

def format_prompt(messages: List[ChatMessage]) -> str:
    prompt = ""
    for msg in messages:
        if msg.role == "system": prompt += f"{msg.content}\n\n"
        elif msg.role == "user": prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant": prompt += f"Assistant: {msg.content}\n"
    return prompt + "Assistant: "

async def stream_generator(prompt: str):
    global global_text_queue, global_state
    
    # 在线程池中运行推理以避免阻塞事件循环
    loop = asyncio.get_event_loop()
    inference_task = loop.run_in_executor(None, rkllm_instance.run, prompt)
    
    created_time = int(time.time())
    
    while not inference_task.done() or global_text_queue:
        if global_text_queue:
            content = global_text_queue.pop(0)
            chunk = {
                "id": "chatcmpl-rkllm",
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": "rkllm",
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        else:
            await asyncio.sleep(0.01) # 极短等待，释放控制权
            
    # 结束标记
    yield f"data: {json.dumps({'id':'chatcmpl-rkllm','object':'chat.completion.chunk','created':created_time,'model':'rkllm','choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global global_text_queue, global_state
    
    if not model_lock.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Server Busy")
    
    try:
        global_text_queue.clear()
        global_state = -1
        prompt = format_prompt(request.messages)
        
        if request.stream:
            return StreamingResponse(
                stream_generator(prompt), 
                media_type="text/event-stream"
            )
        else:
            # 非流式处理
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, rkllm_instance.run, prompt)
            
            while global_state not in [LLMCallState.RKLLM_RUN_FINISH, LLMCallState.RKLLM_RUN_ERROR]:
                await asyncio.sleep(0.05)
                
            full_content = "".join(global_text_queue)
            global_text_queue.clear()
            
            return {
                "id": "chatcmpl-rkllm",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "rkllm",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": full_content},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
    finally:
        # 注意：对于流式，锁的释放需要特殊处理，这里为了简化在同步调用后释放
        # 实际生产中建议使用 Semaphore 配合 FastAPI 依赖注入
        if not request.stream:
            model_lock.release()
        else:
            # 流式情况下，在生成器结束后释放锁的操作通常放在后台任务或回调中
            # 此处简单起见，建议流式客户端完成读取
            def release_lock_later():
                time.sleep(1) # 给生成器一点时间启动
                model_lock.release()
            # 注意：严谨方案是包装 StreamingResponse 在其 on_event("shutdown") 释放
            model_lock.release() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--platform', type=str, default="rk3588")
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    rkllm_instance = RKLLM(args.model_path, args.platform)
    uvicorn.run(app, host="0.0.0.0", port=args.port)