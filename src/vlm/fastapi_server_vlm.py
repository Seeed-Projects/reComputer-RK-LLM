#!/usr/bin/env python3
"""
FastAPI Server for RKLLM Vision Language Model Service
OpenAI-compatible API for multimodal inference on RK3588 platform
"""

import ctypes
import os
import sys
import time
import uuid
import json
import logging
import asyncio
import threading
import argparse
from queue import Queue
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Union, Generator
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ==================== System Library Preloading ====================
def preload_libraries():
    """Preload necessary system libraries to fix OpenCL issues"""
    try:
        # Set environment variables
        os.environ['LD_PRELOAD'] = '/usr/lib/aarch64-linux-gnu/libmali.so.1:' + os.environ.get('LD_PRELOAD', '')
        os.environ['LD_LIBRARY_PATH'] = '/usr/lib/aarch64-linux-gnu:/usr/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

        # Preload libraries
        libs = [
            'libmali.so.1',
            'libOpenCL.so',
            'librknnrt.so',
        ]

        for lib in libs:
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                print(f"✓ Preloaded: {lib}")
            except Exception as e:
                print(f"⚠ Failed to preload {lib}: {e}")
    except Exception as e:
        print(f"⚠ Error during library preloading: {e}")

print("Preloading system libraries...")
preload_libraries()

# ==================== RKLLM Service Wrapper ====================

class RKLLMService:
    """Unified wrapper for RKLLM Vision Language Model service"""
    
    def __init__(self, library_path: str = "/usr/lib/librkllm_service.so"):
        try:
            # Load RKLLM service library
            self.lib = ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
            self._setup_function_signatures()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {library_path}: {e}")
        
        self.ctx = None
        self.lock = threading.Lock()
        self.initialized = False
        self.encoder_model_path = None
        self.llm_model_path = None
        
        # 用于管理流式回调的队列
        self._callback_queues = {}
        self._callback_counter = 0
        
    def _setup_function_signatures(self):
        """Setup C function signatures"""
        # Basic service functions
        self.lib.create_service.restype = ctypes.c_void_p
        self.lib.create_service.argtypes = []
        
        # Initialize without image
        self.lib.initialize_service_without_image.restype = ctypes.c_int
        self.lib.initialize_service_without_image.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p
        ]
        
        # Generate response functions
        self.lib.generate_response_with_params.restype = ctypes.c_char_p
        self.lib.generate_response_with_params.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_float, ctypes.c_float
        ]
        
        # Dynamic image generation
        self.lib.generate_response_with_dynamic_image.restype = ctypes.c_char_p
        self.lib.generate_response_with_dynamic_image.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_float, ctypes.c_float
        ]

        # Define streaming callback function type
        self.streaming_callback_type = ctypes.CFUNCTYPE(
            ctypes.c_int,  # return type
            ctypes.c_char_p,  # token
            ctypes.c_void_p  # userdata
        )

        # Streaming functions with dynamic image
        self.lib.generate_streaming_with_dynamic_image.restype = ctypes.c_int
        self.lib.generate_streaming_with_dynamic_image.argtypes = [
            ctypes.c_void_p,  # ctx
            ctypes.c_char_p,  # image_data_b64
            ctypes.c_size_t,  # image_size
            ctypes.c_char_p,  # prompt
            ctypes.c_int,     # top_k
            ctypes.c_float,   # top_p
            ctypes.c_float,   # temperature
            self.streaming_callback_type,  # callback
            ctypes.c_void_p   # userdata
        ]

        # Streaming functions without image
        self.lib.generate_streaming.restype = ctypes.c_int
        self.lib.generate_streaming.argtypes = [
            ctypes.c_void_p,  # ctx
            ctypes.c_char_p,  # prompt
            ctypes.c_int,     # top_k
            ctypes.c_float,   # top_p
            ctypes.c_float,   # temperature
            self.streaming_callback_type,  # callback
            ctypes.c_void_p   # userdata
        ]

        # Runtime parameter functions
        self.lib.update_runtime_params.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.lib.update_runtime_params.restype = ctypes.c_int

        self.lib.get_runtime_params.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.get_runtime_params.restype = None

        # Cleanup
        self.lib.cleanup_service.argtypes = [ctypes.c_void_p]
        self.lib.destroy_service.argtypes = [ctypes.c_void_p]
    
    def initialize_without_image(self,
                               encoder_model_path: str,
                               llm_model_path: str,
                               max_new_tokens: int = 128,
                               max_context_len: int = 2048,
                               rknn_core_num: int = 1,
                               img_start: str = "<|image_start|>",
                               img_end: str = "<|image_end|>",
                               img_content: str = "<|image_content|>") -> bool:
        """Initialize service without default image"""
        with self.lock:
            if self.ctx:
                self.cleanup()
            
            self.ctx = self.lib.create_service()
            if not self.ctx:
                return False
            
            # Validate model files
            for path in [encoder_model_path, llm_model_path]:
                if not Path(path).exists():
                    self.lib.destroy_service(self.ctx)
                    self.ctx = None
                    return False
            
            self.encoder_model_path = encoder_model_path
            self.llm_model_path = llm_model_path
            
            ret = self.lib.initialize_service_without_image(
                self.ctx,
                encoder_model_path.encode(),
                llm_model_path.encode(),
                max_new_tokens,
                max_context_len,
                rknn_core_num,
                img_start.encode(),
                img_end.encode(),
                img_content.encode()
            )
            
            self.initialized = (ret == 0)
            return self.initialized
    
    def generate(self,
                 prompt: str,
                 top_k: int = 1,
                 top_p: float = 1.0,
                 temperature: float = 0.7) -> str:
        """Generate response for prompt"""
        with self.lock:
            if not self.ctx or not self.initialized:
                return "Error: Service not initialized"
            
            result = self.lib.generate_response_with_params(
                self.ctx,
                prompt.encode(),
                top_k,
                top_p,
                temperature
            )
            
            return result.decode('utf-8', errors='ignore') if result else ""
    
    def generate_with_dynamic_image(self,
                                   image_data_b64: str,
                                   prompt: str,
                                   top_k: int = 1,
                                   top_p: float = 1.0,
                                   temperature: float = 0.7) -> str:
        """Generate response with dynamic image"""
        with self.lock:
            if not self.ctx or not self.initialized:
                return "Error: Service not initialized"
            
            result = self.lib.generate_response_with_dynamic_image(
                self.ctx,
                image_data_b64.encode(),
                len(image_data_b64),
                prompt.encode(),
                top_k,
                top_p,
                temperature
            )
            
            return result.decode('utf-8', errors='ignore') if result else ""

    def _create_streaming_callback(self, callback_id: int) -> ctypes.CFUNCTYPE:
        """创建流式回调函数"""
        @self.streaming_callback_type
        def internal_callback(token_ptr: ctypes.c_char_p, userdata: ctypes.c_void_p) -> ctypes.c_int:
            """内部回调函数，将token放入队列"""
            token = None
            if token_ptr:
                try:
                    # 解码token
                    token_bytes = ctypes.string_at(token_ptr)
                    if token_bytes:
                        token = token_bytes.decode('utf-8', errors='ignore')
                except:
                    token = None
            
            # 将token放入队列
            if callback_id in self._callback_queues:
                self._callback_queues[callback_id].put(token)
            
            return 0  # 成功返回0
        
        return internal_callback

    def generate_streaming_with_dynamic_image_generator(self,
                                                       image_data_b64: str,
                                                       prompt: str,
                                                       top_k: int = 1,
                                                       top_p: float = 1.0,
                                                       temperature: float = 0.7) -> Generator[str, None, None]:
        """生成流式响应的生成器（带动态图像）"""
        with self.lock:
            if not self.ctx or not self.initialized:
                raise RuntimeError("Service not initialized")
            
            # 创建回调队列
            callback_id = self._callback_counter
            self._callback_counter += 1
            callback_queue = Queue()
            self._callback_queues[callback_id] = callback_queue
            
            # 创建回调函数
            c_callback = self._create_streaming_callback(callback_id)
            
            # 创建线程来运行流式推理
            def run_streaming():
                try:
                    result = self.lib.generate_streaming_with_dynamic_image(
                        self.ctx,
                        image_data_b64.encode(),
                        len(image_data_b64),
                        prompt.encode(),
                        top_k,
                        top_p,
                        temperature,
                        c_callback,
                        None  # userdata
                    )
                    
                    if result != 0:
                        callback_queue.put(None)  # 发送结束信号
                except Exception as e:
                    print(f"Streaming error: {e}")
                    callback_queue.put(None)  # 发送结束信号
                finally:
                    # 确保发送结束信号
                    if callback_id in self._callback_queues:
                        self._callback_queues[callback_id].put(None)
            
            # 启动流式推理线程
            streaming_thread = threading.Thread(target=run_streaming)
            streaming_thread.daemon = True
            streaming_thread.start()
            
            # 从队列中获取token并生成
            try:
                while True:
                    token = callback_queue.get(timeout=300)  # 5分钟超时
                    if token is None:  # 结束信号
                        break
                    if token:  # 有效的token
                        yield token
            except Exception as e:
                print(f"Queue error: {e}")
            finally:
                # 清理
                if callback_id in self._callback_queues:
                    del self._callback_queues[callback_id]
                # 等待线程结束
                streaming_thread.join(timeout=1)

    def generate_streaming_generator(self,
                                    prompt: str,
                                    top_k: int = 1,
                                    top_p: float = 1.0,
                                    temperature: float = 0.7) -> Generator[str, None, None]:
        """生成流式响应的生成器（不带图像）"""
        with self.lock:
            if not self.ctx or not self.initialized:
                raise RuntimeError("Service not initialized")
            
            # 创建回调队列
            callback_id = self._callback_counter
            self._callback_counter += 1
            callback_queue = Queue()
            self._callback_queues[callback_id] = callback_queue
            
            # 创建回调函数
            c_callback = self._create_streaming_callback(callback_id)
            
            # 创建线程来运行流式推理
            def run_streaming():
                try:
                    result = self.lib.generate_streaming(
                        self.ctx,
                        prompt.encode(),
                        top_k,
                        top_p,
                        temperature,
                        c_callback,
                        None  # userdata
                    )
                    
                    if result != 0:
                        callback_queue.put(None)  # 发送结束信号
                except Exception as e:
                    print(f"Streaming error: {e}")
                    callback_queue.put(None)  # 发送结束信号
                finally:
                    # 确保发送结束信号
                    if callback_id in self._callback_queues:
                        self._callback_queues[callback_id].put(None)
            
            # 启动流式推理线程
            streaming_thread = threading.Thread(target=run_streaming)
            streaming_thread.daemon = True
            streaming_thread.start()
            
            # 从队列中获取token并生成
            try:
                while True:
                    token = callback_queue.get(timeout=300)  # 5分钟超时
                    if token is None:  # 结束信号
                        break
                    if token:  # 有效的token
                        yield token
            except Exception as e:
                print(f"Queue error: {e}")
            finally:
                # 清理
                if callback_id in self._callback_queues:
                    del self._callback_queues[callback_id]
                # 等待线程结束
                streaming_thread.join(timeout=1)

    def update_runtime_params(self, max_new_tokens=None, max_context_len=None, rknn_core_num=None):
        """Update runtime parameters"""
        with self.lock:
            if not self.ctx or not self.initialized:
                return False

            ret = self.lib.update_runtime_params(
                self.ctx,
                max_new_tokens if max_new_tokens is not None else 128,
                max_context_len if max_context_len is not None else 2048,
                rknn_core_num if rknn_core_num is not None else 1
            )

            return ret == 0

    def get_runtime_params(self):
        """Get current runtime parameters"""
        with self.lock:
            if not self.ctx or not self.initialized:
                return {}

            max_tokens = ctypes.c_int()
            max_context = ctypes.c_int()
            rknn_cores = ctypes.c_int()

            self.lib.get_runtime_params(
                self.ctx,
                ctypes.byref(max_tokens),
                ctypes.byref(max_context),
                ctypes.byref(rknn_cores)
            )

            return {
                'max_new_tokens': max_tokens.value,
                'max_context_len': max_context.value,
                'rknn_core_num': rknn_cores.value
            }

    def get_service_info(self):
        """Get service information"""
        return {
            "initialized": self.initialized,
            "encoder_model": self.encoder_model_path,
            "llm_model": self.llm_model_path,
            "runtime_params": self.get_runtime_params(),
            "capabilities": {
                "streaming": True,  # 我们总是支持流式
                "runtime_params": True
            }
        }

    def cleanup(self):
        """Clean up resources"""
        with self.lock:
            if self.ctx:
                self.lib.cleanup_service(self.ctx)
                self.initialized = False
            # 清理所有回调队列
            self._callback_queues.clear()

    def __del__(self):
        """Destructor"""
        if hasattr(self, 'ctx') and self.ctx:
            self.lib.destroy_service(self.ctx)

# ==================== Pydantic Model Definitions ====================
class ImageUrl(BaseModel):
    url: str = Field(..., description="Image URL or base64 encoded image data")
    detail: Optional[str] = Field(default="auto", description="Detail level for the image")

class ContentPart(BaseModel):
    type: str = Field(..., description="Content type: text or image_url")
    text: Optional[str] = Field(None, description="Text content")
    image_url: Optional[ImageUrl] = Field(None, description="Image URL")

class Message(BaseModel):
    role: str = Field(..., description="Message role: system, user, assistant")
    content: Union[str, List[ContentPart]] = Field(..., description="Message content")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="rkllm-vision", description="Model name")
    messages: List[Message] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Temperature parameter (0.0-2.0)")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter (0.0-1.0)")
    top_k: Optional[int] = Field(default=1, ge=1, le=100, description="Top-k sampling parameter (1-100)")
    n: Optional[int] = Field(default=1, ge=1, le=10, description="Number of completions to generate")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response")
    max_tokens: Optional[int] = Field(default=50, ge=1, le=4096, description="Maximum tokens to generate")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    max_context_len: Optional[int] = Field(default=2048, ge=512, le=8192, description="Maximum context length")
    rknn_core_num: Optional[int] = Field(default=3, ge=1, le=4, description="Number of RKNN cores to use")

class UsageInfo(BaseModel):
    prompt_tokens: int = Field(default=0, description="Prompt tokens")
    completion_tokens: int = Field(default=0, description="Completion tokens")
    total_tokens: int = Field(default=0, description="Total tokens")

class ChatCompletionResponseChoice(BaseModel):
    index: int = Field(..., description="Choice index")
    message: Message = Field(..., description="Message")
    finish_reason: Optional[str] = Field(default="stop", description="Finish reason")

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Request ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    choices: List[ChatCompletionResponseChoice] = Field(..., description="List of choices")
    usage: UsageInfo = Field(..., description="Usage information")
    system_fingerprint: Optional[str] = Field(default="fp_rkllm_vision", description="System fingerprint")

class DeltaMessage(BaseModel):
    role: Optional[str] = Field(None, description="Role")
    content: Optional[str] = Field(None, description="Content")

class ChatCompletionStreamResponseChoice(BaseModel):
    index: int = Field(..., description="Choice index")
    delta: DeltaMessage = Field(..., description="Delta message")
    finish_reason: Optional[str] = Field(None, description="Finish reason")

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(..., description="Request ID")
    object: str = Field(default="chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    choices: List[ChatCompletionStreamResponseChoice] = Field(..., description="List of choices")
    system_fingerprint: Optional[str] = Field(default="fp_rkllm_vision", description="System fingerprint")

# ==================== Server Configuration ====================
class ServerConfig:
    def __init__(self):
        self.max_context_len = 2048
        self.default_temperature = 0.7
        self.default_top_p = 1.0
        self.default_top_k = 1
        self.default_max_tokens = 50
        self.max_concurrent_requests = 1
        self.timeout_seconds = 300
        self.rknn_core_num = 3
        self.encoder_model_path = ""
        self.llm_model_path = ""
        self.img_start = "<|image_start|>"
        self.img_end = "<|image_end|>"
        self.img_content = "<|image_content|>"
        self.library_path = "/usr/lib/librkllm_service.so"

config = ServerConfig()

# Global variables
request_lock = threading.Lock()
active_requests = 0
rkllm_service = None
executor = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Helper Functions ====================
def extract_user_content(messages: List[Message]) -> tuple[str, str]:
    """提取用户内容和图片数据"""
    import base64
    import requests
    from io import BytesIO
    from PIL import Image
    
    text_content = ""
    image_data_b64 = None
    
    for msg in messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                text_content = msg.content
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if item.type == "text" and item.text:
                        text_content = item.text
                    elif item.type == "image_url" and item.image_url:
                        url = item.image_url.url
                        
                        # 处理base64格式
                        if url.startswith("data:image"):
                            parts = url.split(",")
                            if len(parts) > 1:
                                image_data_b64 = parts[1]
                            else:
                                image_data_b64 = url
                        # 处理HTTP/HTTPS URL - 新增
                        elif url.startswith(("http://", "https://")):
                            try:
                                logger.info(f"下载图片: {url}")
                                response = requests.get(url, timeout=30)
                                response.raise_for_status()
                                
                                # 验证是否为图片
                                content_type = response.headers.get('content-type', '')
                                if not content_type.startswith('image/'):
                                    logger.warning(f"URL返回的不是图片: {content_type}")
                                
                                # 转换为base64
                                image_bytes = response.content
                                image_data_b64 = base64.b64encode(image_bytes).decode('utf-8')
                                logger.info(f"图片下载成功，base64长度: {len(image_data_b64)}")
                                
                            except Exception as e:
                                logger.error(f"下载图片失败 {url}: {e}")
                                image_data_b64 = None
                        else:
                            # 可能是文件路径或其他
                            logger.warning(f"无法识别的图片格式: {url}")
                            image_data_b64 = url
            break
    
    return text_content, image_data_b64

def estimate_tokens(text: str) -> int:
    """Estimate token count"""
    if not text:
        return 0
    return int(len(text) * 0.3)

# ==================== Application Lifecycle Management ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    global rkllm_service, executor
    
    logger.info("=" * 60)
    logger.info("Starting RKLLM Vision API Server")
    logger.info("=" * 60)
    
    # Initialize thread pool
    executor = ThreadPoolExecutor(
        max_workers=config.max_concurrent_requests + 1,
        thread_name_prefix="rkllm_vision_worker"
    )
    logger.info("✅ Thread pool initialized")
    
    # Initialize RKLLM service
    try:
        lib_path = config.library_path
        
        rkllm_service = RKLLMService(lib_path)
        
        success = rkllm_service.initialize_without_image(
            encoder_model_path=config.encoder_model_path,
            llm_model_path=config.llm_model_path,
            max_new_tokens=config.default_max_tokens,
            max_context_len=config.max_context_len,
            rknn_core_num=config.rknn_core_num,
            img_start=config.img_start,
            img_end=config.img_end,
            img_content=config.img_content
        )
        
        if not success:
            raise RuntimeError("Failed to initialize RKLLM vision service")
        
        logger.info("✅ RKLLM vision service initialized successfully!")
        logger.info(f"  Vision encoder: {Path(config.encoder_model_path).name}")
        logger.info(f"  LLM model: {Path(config.llm_model_path).name}")
        
        service_info = rkllm_service.get_service_info()
        logger.info(f"  Streaming support: {'Yes' if service_info['capabilities']['streaming'] else 'No'}")
        logger.info(f"  Runtime params support: {'Yes' if service_info['capabilities']['runtime_params'] else 'No'}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("\nShutting down server...")
    
    if executor:
        executor.shutdown(wait=False)
        logger.info("✅ Thread pool shut down")
    
    if rkllm_service:
        rkllm_service.cleanup()
        logger.info("✅ Service resources released")

# ==================== FastAPI Application ====================
app = FastAPI(
    title="RKLLM Vision API Server",
    version="1.0.0",
    description="OpenAI API compatible server for RKLLM Vision Language Models",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== API Endpoints ====================
@app.get("/")
async def root():
    """Root endpoint"""
    service_info = rkllm_service.get_service_info() if rkllm_service else None
    
    return {
        "message": "RKLLM Vision API Server",
        "status": "running",
        "service": {
            "initialized": service_info["initialized"] if service_info else False,
            "vision_model": Path(config.encoder_model_path).name if config.encoder_model_path else None,
            "llm_model": Path(config.llm_model_path).name if config.llm_model_path else None,
        },
        "version": "1.0.0",
        "endpoints": {
            "GET /": "Server information",
            "GET /health": "Health check",
            "GET /v1/models": "List models",
            "POST /v1/chat/completions": "Chat completion"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    service_info = rkllm_service.get_service_info() if rkllm_service else None
    
    return {
        "status": "healthy" if rkllm_service and service_info["initialized"] else "unhealthy",
        "service_initialized": service_info["initialized"] if service_info else False,
        "active_requests": active_requests,
        "max_concurrent": config.max_concurrent_requests,
        "timestamp": int(time.time())
    }

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "rkllm-vision",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rockchip"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion"""
    global active_requests
    
    with request_lock:
        if active_requests >= config.max_concurrent_requests:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "message": "Too many requests, please try again later",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded"
                    }
                }
            )
        active_requests += 1
    
    try:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        
        logger.info(f"[{request_id}] New request: stream={request.stream}, messages={len(request.messages)}")
        
        # Extract content
        text_content, image_b64 = extract_user_content(request.messages)
        
        if not text_content:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Prepare prompt - IMPORTANT: Add <image> tag if there's an image
        if image_b64:
            prompt = f"<image>{text_content}"
        else:
            prompt = text_content
        
        # Update runtime parameters if needed
        if (request.max_tokens != config.default_max_tokens or 
            request.max_context_len != config.max_context_len or
            request.rknn_core_num != config.rknn_core_num):
            
            success = rkllm_service.update_runtime_params(
                max_new_tokens=request.max_tokens,
                max_context_len=request.max_context_len,
                rknn_core_num=request.rknn_core_num
            )
            if not success:
                logger.warning(f"[{request_id}] Failed to update runtime parameters")
        
        logger.info(f"[{request_id}] Processing request with {'image' if image_b64 else 'text only'}")
        
        if request.stream:
            # 真正的流式响应
            async def generate_real_stream():
                try:
                    # 发送初始chunk（assistant角色）
                    initial_chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamResponseChoice(
                                index=0,
                                delta=DeltaMessage(role="assistant"),
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {initial_chunk.model_dump_json(exclude_unset=True, ensure_ascii=False)}\n\n"
                    
                    # 创建流式生成器
                    if image_b64:
                        # 使用真正的流式推理
                        def streaming_task():
                            try:
                                generator = rkllm_service.generate_streaming_with_dynamic_image_generator(
                                    image_data_b64=image_b64,
                                    prompt=prompt,
                                    top_k=request.top_k,
                                    top_p=request.top_p,
                                    temperature=request.temperature
                                )
                                result = []
                                for token in generator:
                                    result.append(token)
                                return "".join(result)
                            except Exception as e:
                                logger.error(f"[{request_id}] Streaming error: {e}")
                                return f"[ERROR] {e}"
                    else:
                        # 纯文本流式推理
                        def streaming_task():
                            try:
                                generator = rkllm_service.generate_streaming_generator(
                                    prompt=prompt,
                                    top_k=request.top_k,
                                    top_p=request.top_p,
                                    temperature=request.temperature
                                )
                                result = []
                                for token in generator:
                                    result.append(token)
                                return "".join(result)
                            except Exception as e:
                                logger.error(f"[{request_id}] Streaming error: {e}")
                                return f"[ERROR] {e}"
                    
                    # 在单独的线程中运行流式推理
                    future = executor.submit(streaming_task)
                    
                    # 我们需要实时获取token，但上面的实现是收集所有token后再返回
                    # 为了真正的流式，我们需要修改生成器以实时回调
                    # 这里我们使用一个队列来实现实时token传递
                    token_queue = Queue()
                    
                    # 定义实时回调的生成器
                    def real_time_generator():
                        if image_b64:
                            for token in rkllm_service.generate_streaming_with_dynamic_image_generator(
                                image_data_b64=image_b64,
                                prompt=prompt,
                                top_k=request.top_k,
                                top_p=request.top_p,
                                temperature=request.temperature
                            ):
                                token_queue.put(token)
                        else:
                            for token in rkllm_service.generate_streaming_generator(
                                prompt=prompt,
                                top_k=request.top_k,
                                top_p=request.top_p,
                                temperature=request.temperature
                            ):
                                token_queue.put(token)
                        token_queue.put(None)  # 结束信号
                    
                    # 启动实时生成器线程
                    gen_thread = threading.Thread(target=real_time_generator)
                    gen_thread.daemon = True
                    gen_thread.start()
                    
                    # 从队列中获取token并发送
                    while True:
                        try:
                            token = token_queue.get(timeout=300)  # 5分钟超时
                            if token is None:  # 结束信号
                                break
                            
                            if token:
                                # 发送token chunk
                                data_chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    created=created,
                                    model=request.model,
                                    choices=[
                                        ChatCompletionStreamResponseChoice(
                                            index=0,
                                            delta=DeltaMessage(content=token),
                                            finish_reason=None
                                        )
                                    ]
                                )
                                yield f"data: {data_chunk.model_dump_json(exclude_unset=True, ensure_ascii=False)}\n\n"
                        
                        except Exception as e:
                            logger.error(f"[{request_id}] Token queue error: {e}")
                            break
                    
                    # 发送完成chunk
                    done_chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamResponseChoice(
                                index=0,
                                delta=DeltaMessage(),
                                finish_reason="stop"
                            )
                        ]
                    )
                    yield f"data: {done_chunk.model_dump_json(exclude_unset=True, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    logger.error(f"[{request_id}] Stream generation error: {e}")
                    error_data = {"error": {"message": str(e)}}
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            return StreamingResponse(
                generate_real_stream(),
                media_type="text/event-stream"
            )
        else:
            # 非流式响应
            def process_inference():
                """Process inference in thread pool"""
                try:
                    if image_b64:
                        response_text = rkllm_service.generate_with_dynamic_image(
                            image_data_b64=image_b64,
                            prompt=prompt,
                            top_k=request.top_k,
                            top_p=request.top_p,
                            temperature=request.temperature
                        )
                    else:
                        response_text = rkllm_service.generate(
                            prompt=prompt,
                            top_k=request.top_k,
                            top_p=request.top_p,
                            temperature=request.temperature
                        )
                    
                    return response_text
                except Exception as e:
                    logger.error(f"[{request_id}] Inference error: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            try:
                response_text = await asyncio.get_event_loop().run_in_executor(
                    executor, process_inference
                )
                
                # Estimate token usage
                prompt_tokens = estimate_tokens(text_content)
                completion_tokens = estimate_tokens(response_text)
                
                # Build response
                response = ChatCompletionResponse(
                    id=request_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionResponseChoice(
                            index=0,
                            message=Message(
                                role="assistant",
                                content=response_text
                            ),
                            finish_reason="stop"
                        )
                    ],
                    usage=UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens
                    )
                )
                
                return response
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
    finally:
        with request_lock:
            active_requests -= 1

# ==================== Main Program ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RKLLM Vision API Compatible Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rkllm_vision_server.py \\
    --encoder_model ../model/Qwen2-VL-2B_VISION_RK3588.rknn \\
    --llm_model ../model/Qwen2-VL-2B_LLM_W8A8_RK3588.rkllm
  
  python rkllm_vision_server.py \\
    --encoder_model ../model/vision.rknn \\
    --llm_model ../model/llm.rkllm \\
    --port 8080 --max_concurrent 1 --default_max_tokens 50
        """
    )
    
    parser.add_argument('--encoder_model', type=str, required=True,
                       help='Path to vision encoder model (.rknn)')
    parser.add_argument('--llm_model', type=str, required=True,
                       help='Path to LLM model (.rkllm)')
    
    parser.add_argument('--port', type=int, default=8002,
                       help='Server port (default: 8002)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Server host (default: 0.0.0.0)')
    
    parser.add_argument('--max_context_len', type=int, default=2048,
                       help='Maximum context length (default: 2048)')
    parser.add_argument('--default_temperature', type=float, default=0.7,
                       help='Default temperature parameter (default: 0.7)')
    parser.add_argument('--default_top_p', type=float, default=1.0,
                       help='Default top_p parameter (default: 1.0)')
    parser.add_argument('--default_top_k', type=int, default=1,
                       help='Default top_k parameter (default: 1)')
    parser.add_argument('--default_max_tokens', type=int, default=50,
                       help='Default maximum tokens to generate (default: 50)')
    
    parser.add_argument('--max_concurrent', type=int, default=1,
                       help='Maximum concurrent requests (default: 1)')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Request timeout in seconds (default: 300)')
    parser.add_argument('--rknn_core_num', type=int, default=3,
                       help='Number of RKNN cores to use (default: 3)')
    
    parser.add_argument('--img_start', type=str, default='<|image_start|>',
                       help='Image start token (default: <|image_start|>)')
    parser.add_argument('--img_end', type=str, default='<|image_end|>',
                       help='Image end token (default: <|image_end|>)')
    parser.add_argument('--img_content', type=str, default='<|image_content|>',
                       help='Image content token (default: <|image_content|>)')
    
    parser.add_argument('--library_path', type=str, default='/usr/lib/librkllm_service.so',
                       help='Path to RKLLM service library')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Validate files
    for path, name in [(args.encoder_model, "encoder model"),
                      (args.llm_model, "LLM model")]:
        if not os.path.exists(path):
            print(f"❌ Error: {name} file not found: {path}")
            sys.exit(1)
    
    if not os.path.exists(args.library_path):
        print(f"❌ Error: RKLLM service library not found: {args.library_path}")
        sys.exit(1)
    
    # Apply configuration
    config.encoder_model_path = os.path.abspath(args.encoder_model)
    config.llm_model_path = os.path.abspath(args.llm_model)
    config.library_path = os.path.abspath(args.library_path)
    config.max_context_len = args.max_context_len
    config.default_temperature = args.default_temperature
    config.default_top_p = args.default_top_p
    config.default_top_k = args.default_top_k
    config.default_max_tokens = args.default_max_tokens
    config.max_concurrent_requests = args.max_concurrent
    config.timeout_seconds = args.timeout
    config.rknn_core_num = args.rknn_core_num
    config.img_start = args.img_start
    config.img_end = args.img_end
    config.img_content = args.img_content
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Print configuration
    print("=" * 60)
    print("Configuration:")
    print(f"  Vision encoder: {Path(config.encoder_model_path).name}")
    print(f"  LLM model: {Path(config.llm_model_path).name}")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Max context length: {config.max_context_len}")
    print(f"  Default temperature: {config.default_temperature}")
    print(f"  Default Top-p: {config.default_top_p}")
    print(f"  Default Top-k: {config.default_top_k}")
    print(f"  Default max tokens: {config.default_max_tokens}")
    print(f"  RKNN cores: {config.rknn_core_num}")
    print(f"  Max concurrent requests: {config.max_concurrent_requests}")
    print(f"  Request timeout: {config.timeout_seconds}s")
    print("=" * 60)
    
    # Start server
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="debug" if args.debug else "info",
            access_log=True,
            timeout_keep_alive=config.timeout_seconds
        )
    except KeyboardInterrupt:
        print("\n👋 Server interrupted by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)