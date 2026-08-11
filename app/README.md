# reComputer-RK-LLM runtime applications

The two API servers in this directory are copied into the shared environment
image. They load the official Rockchip libraries from `/usr/lib` and expose
OpenAI-compatible endpoints on port `8001`.

- `fastapi_server_llm.py`: text-only RKLLM server
- `fastapi_server_vlm.py`: RKNN vision encoder plus multimodal RKLLM server
