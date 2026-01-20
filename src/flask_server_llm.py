import ctypes
import sys
import os
import threading
import time
import argparse
from flask import Flask, request, jsonify, Response
import json

app = Flask(__name__)

# 设置动态库路径
rkllm_lib = ctypes.CDLL('/usr/lib/librkllmrt.so')

# 定义结构体
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL  = 0
LLMCallState.RKLLM_RUN_WAITING  = 1
LLMCallState.RKLLM_RUN_FINISH  = 2
LLMCallState.RKLLM_RUN_ERROR   = 3

RKLLMInputType = ctypes.c_int
RKLLMInputType.RKLLM_INPUT_PROMPT      = 0

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0

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
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", RKLLMInputType),
        ("input_data", RKLLMInputUnion)
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
    ]

# 锁和状态变量
lock = threading.Lock()
is_blocking = False

# 回调函数输出
global_text = []
global_state = -1

# 回调函数
def callback_impl(result, userdata, state):
    global global_text, global_state
    if state == LLMCallState.RKLLM_RUN_FINISH:
        global_state = state
        print("\n")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        global_state = state
        print("run error")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_NORMAL:
        global_state = state
        if result.contents.text:
            global_text.append(result.contents.text.decode('utf-8'))
    return 0

callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)

# RKLLM 类
class RKLLM(object):
    def __init__(self, model_path, platform="rk3588"):
        # 初始化 RKLLMParam
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')
        rkllm_param.max_context_len = 4096
        rkllm_param.max_new_tokens = 4096
        rkllm_param.n_keep = 0
        rkllm_param.top_k = 1
        rkllm_param.top_p = 0.9
        rkllm_param.temperature = 0.8
        rkllm_param.repeat_penalty = 1.1
        rkllm_param.frequency_penalty = 0.0
        rkllm_param.presence_penalty = 0.0
        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1
        rkllm_param.skip_special_token = True
        rkllm_param.is_async = False
        rkllm_param.img_start = "".encode('utf-8')
        rkllm_param.img_end = "".encode('utf-8')
        rkllm_param.img_content = "".encode('utf-8')
        
        # 设置 extend_param
        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.n_batch = 1  # 关键修复：正确设置 n_batch
        rkllm_param.extend_param.use_cross_attn = 0
        rkllm_param.extend_param.enabled_cpus_num = 4
        
        # 根据平台设置 CPU 掩码
        if platform.lower() in ["rk3576", "rk3588"]:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
        else:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)

        self.handle = RKLLM_Handle_t()

        # 初始化函数
        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int
        
        ret = self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback)
        if ret != 0:
            print("rkllm init failed")
            sys.exit(1)
        else:
            print("rkllm init success!")

        # 运行函数
        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int
        
        # 销毁函数
        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

    def run(self, prompt, role="user"):
        rkllm_input = RKLLMInput()
        rkllm_input.role = role.encode('utf-8')
        rkllm_input.enable_thinking = False
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode('utf-8'))
        
        infer_param = RKLLMInferParam()
        infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        infer_param.lora_params = None
        infer_param.prompt_cache_params = None
        infer_param.keep_history = 0
        
        return self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(infer_param), None)
    
    def release(self):
        self.rkllm_destroy(self.handle)

# Flask 路由
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    global global_text, global_state, is_blocking
    
    if is_blocking:
        return jsonify({'error': {'message': 'Server is busy', 'type': 'server_error'}}), 503
    
    lock.acquire()
    try:
        is_blocking = True
        
        data = request.json
        if not data or 'messages' not in data:
            return jsonify({'error': {'message': 'Invalid request', 'type': 'invalid_request'}}), 400
        
        messages = data['messages']
        stream = data.get('stream', False)
        
        # 重置全局变量
        global_text = []
        global_state = -1
        
        # 构建提示词
        prompt = ""
        for msg in messages:
            if msg['role'] == 'system':
                prompt += f"{msg['content']}\n\n"
            elif msg['role'] == 'user':
                prompt += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                prompt += f"Assistant: {msg['content']}\n"
        
        # 添加最后的 Assistant: 提示
        if prompt and not prompt.endswith("Assistant: "):
            prompt += "Assistant: "
        
        print(f"Prompt: {prompt}")
        
        def generate_response():
            nonlocal prompt
            
            # 运行模型
            model_thread = threading.Thread(target=rkllm_model.run, args=(prompt,))
            model_thread.start()
            
            if stream:
                # 流式响应 - 直接返回纯文本
                model_thread_finished = False
                while not model_thread_finished:
                    if global_text:
                        chunk = global_text.pop(0)
                        yield chunk  # 直接返回文本内容，不包装为JSON
                    
                    model_thread.join(timeout=0.01)
                    model_thread_finished = not model_thread.is_alive()
                    
                    if global_state == LLMCallState.RKLLM_RUN_FINISH:
                        break
                
                # 添加结束标记
                yield "[DONE]"
            else:
                # 非流式响应 - 返回JSON
                model_thread_finished = False
                full_response = ""
                while not model_thread_finished:
                    while global_text:
                        full_response += global_text.pop(0)
                    
                    model_thread.join(timeout=0.01)
                    model_thread_finished = not model_thread.is_alive()
                    
                    if global_state == LLMCallState.RKLLM_RUN_FINISH:
                        break
                
                response = {
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "rkllm-model",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_response
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
                return json.dumps(response, ensure_ascii=False)
        
        if stream:
            # 对于流式响应，返回纯文本流
            return Response(generate_response(), content_type='text/plain; charset=utf-8')
        else:
            # 对于非流式响应，返回JSON
            response_data = generate_response()
            return Response(response_data, content_type='application/json; charset=utf-8')
            
    finally:
        lock.release()
        is_blocking = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', type=str, required=True, 
                       help='Absolute path of the converted RKLLM model')
    parser.add_argument('--target_platform', type=str, required=True,
                       help='Target platform: e.g., rk3588/rk3576')
    parser.add_argument('--port', type=int, default=8080,
                       help='Port to run the server on')
    args = parser.parse_args()

    if not os.path.exists(args.rkllm_model_path):
        print(f"Error: Model path does not exist: {args.rkllm_model_path}")
        sys.exit(1)

    # 初始化模型
    print("Initializing RKLLM model...")
    rkllm_model = RKLLM(args.rkllm_model_path, args.target_platform)
    print("Model initialized successfully!")
    
    # 启动服务器
    print(f"Starting server on port {args.port}...")
    app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)
    
    # 清理
    rkllm_model.release()
    print("Server stopped.")