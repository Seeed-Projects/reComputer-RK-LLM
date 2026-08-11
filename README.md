# RKLLM Docker

Docker images for running pre-converted RKLLM LLM and VLM models on Rockchip
RK3576 and RK3588-family boards through OpenAI- and Ollama-compatible APIs.

The project now follows a two-layer image design:

```text
runtime/ + app/ + scripts/  ->  environment image  ->  model image
```

The environment image contains the pinned ARM64 RKLLM/RKNN runtimes and API
servers. Model images add only the selected `.rkllm` and optional `.rknn`
artifacts. Custom models can use the environment image directly by mounting
their files.

Deployment guides: [LLM](docs/LLM.md) · [VLM](docs/VLM.md)

## Runtime

The bundled RKLLM runtime is v1.3.0. The ctypes definitions in
`app/fastapi_server_llm.py` and `app/fastapi_server_vlm.py` match that ABI.
Runtime artifacts are kept under [`runtime/`](runtime/).

## Quick start

Build the environment image for the target board:

```bash
docker buildx build --platform linux/arm64 \
  -f docker/Dockerfile \
  -t rkllm:env --load .
```

Run a custom LLM model:

```bash
sudo docker run --rm -d \
  --name rkllm \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  -v ./models:/app/models:ro \
  -e MODEL_FILE=Qwen2.5-1.5B-Instruct_w8a8_RK3576.rkllm \
  -e TARGET_PLATFORM=rk3576 \
  rkllm:env
```

Check readiness and call the API:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rkllm-model","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

For an image containing a model, build the shared environment first and then
use `docker/Dockerfile.model` with `models/.build/` populated. The declarative
model definitions under `models/llm/` and `models/vlm/` are consumed by the
GitHub Actions workflow.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `/app/models/model.rkllm` | Full model path |
| `MODEL_FILE` | empty | Model filename under `/app/models` |
| `MODEL_KIND` | `llm` | `llm` or `vlm` |
| `VISION_MODEL_FILE` | empty | VLM `.rknn` filename |
| `TARGET_PLATFORM` | `auto` | `rk3576`, `rk3588`, or `rk3588s` |
| `RUN_FREQ_FIX` | `true` | Apply board frequency setup |
| `PORT` | `8001` | HTTP port |
| `API_MODEL_NAME` | `rkllm-model` | Public LLM model name |
| `INTERACTIVE_CHAT` | `false` | Enable terminal chat for LLM |

## Repository layout

```text
app/                         API servers
runtime/lib/                 ARM64 RKLLM/RKNN shared libraries
runtime/wheels/              ARM64 Python runtime wheels
models/<kind>/<id>/<board>/  Model metadata and download URLs
docker/Dockerfile            Reusable environment image
docker/Dockerfile.model      Thin model image
docker/entrypoint.sh         Runtime/model selection and validation
scripts/                     Board frequency setup
.github/workflows/           Consolidated environment/model builds
docs/LLM.md                  LLM deployment guide
docs/VLM.md                  VLM deployment guide
tools/                       API performance test clients
```

## Safety

The examples use `--privileged` and `-v /dev:/dev` because Rockchip NPU access
varies by board image. Keep the service on a trusted network; authentication
and TLS are not included.

## References

- [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm)
- [Hanzo-Huang/rkllm-docker](https://github.com/Hanzo-Huang/rkllm-docker/)
