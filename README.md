[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

# reComputer-RK-LLM

Docker-based deployment for pre-converted large language models (LLMs) and
vision-language models (VLMs) on Seeed Studio reComputer boards with Rockchip
RK3576 and RK3588-family processors. The bundled servers expose
OpenAI-compatible APIs, with an Ollama-compatible chat endpoint for LLMs.

## Hardware

This project targets reComputer RK3576 and reComputer RK3588 boards with a
64-bit Linux image, Docker, and Docker Buildx installed. The repository
includes the ARM64 runtime libraries and Python wheel required by the image.
The containers need access to the board's NPU devices.

## Deployment guides

- [LLM deployment](docs/LLM.md)
- [VLM deployment](docs/VLM.md)

## Architecture

The repository builds a reusable ARM64 environment image and separate model
images:

```text
runtime/ + app/ + scripts/  ->  environment image  ->  model image
```

The environment image contains the pinned RKLLM/RKNN runtimes and API servers.
Model images add only the selected `.rkllm` and optional `.rknn` artifacts.
Custom models can use the environment image directly by mounting their files.

## Quick start

Run a published model image. This example starts the Qwen2.5 1.5B Instruct
model for RK3576 with the `w8a8` quantization:

```bash
sudo docker run --rm -it \
  --name recomputer-rk-llm \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  -e INTERACTIVE_CHAT=true \
  -e LOG_LEVEL=warning \
  ghcr.io/seeed-projects/recomputer-rk-llm/llm/qwen2.5-1.5b-instruct:rk3576-w8a8
```

The command stays attached to the terminal and enables interactive chat for
testing. If you want to run the same service in the background, use this
command instead:

```bash
sudo docker run --rm -d \
  --name recomputer-rk-llm \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  ghcr.io/seeed-projects/recomputer-rk-llm/llm/qwen2.5-1.5b-instruct:rk3576-w8a8
```

Choose another published image from [Available model definitions](#available-model-definitions).
The image already contains the converted model and its matching runtime; use
the [LLM guide](docs/LLM.md) or [VLM guide](docs/VLM.md) for custom model
files and request examples.

Check readiness and call the OpenAI-compatible API:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rkllm-model","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

For a custom VLM, set `MODEL_KIND=vlm` and provide both `MODEL_FILE` and
`VISION_MODEL_FILE`; see the [VLM guide](docs/VLM.md) for the complete request
format.

## Build locally

To build the shared environment image on the target board or with an ARM64
Buildx builder:

```bash
docker buildx build --platform linux/arm64 \
  -f docker/Dockerfile \
  -t recomputer-rk-llm:env --load .
```

## Available model definitions

The definitions under [`models/`](models/) are used by the **Build model
images** GitHub Actions workflow. Each row below is a separate model image;
the values show the available quantization tags for each board:

| Type | Model | RK3576 | RK3588 |
| --- | --- | --- | --- |
| LLM | [Qwen2.5 1.5B Instruct](models/llm/qwen2.5-1.5b-instruct) | `w4a16`, `w8a8` | `w8a8` |
| LLM | [Qwen2.5 3B Instruct](models/llm/qwen2.5-3b-instruct) | `w4a16`, `w8a8` | `w8a8` |
| LLM | [Qwen2 0.5B Instruct](models/llm/qwen2-0.5b-instruct) | `w4a16-g128`, `w8a8` | `w8a8` |
| LLM | [Qwen3 1.7B](models/llm/qwen3-1.7b) | `w4a16`, `w4a16-g128`, `w8a8` | `w8a8` |
| LLM | [Qwen3 4B](models/llm/qwen3-4b) | `w4a16`, `w4a16-g128`, `w8a8` | `w8a8` |
| LLM | [DeepSeek R1 Distill Qwen 1.5B](models/llm/deepseek-r1-distill-qwen-1.5b) | `w4a16-g128`, `w8a8` | `w8a8` |
| LLM | [Gemma 4 E2B IT](models/llm/gemma-4-e2b-it) | `w4a16-g128`, `w8a8` | `w8a8` |
| LLM | [Gemma 3 4B IT](models/llm/gemma-3-4b-it) | `w4a16`, `w8a8` | `w8a8` |
| LLM | [MiniCPM3 4B](models/llm/minicpm3-4b) | `w4a16-g128`, `w8a8` | `w8a8` |
| LLM | [MiniCPM4 0.5B](models/llm/minicpm4-0.5b) | `w4a16-g128`, `w8a8` | `w8a8` |
| LLM | [Llama 3.2 1B Instruct](models/llm/llama-3.2-1b-instruct) | `w4a16-g128`, `w8a8` | `w8a8` |
| LLM | [Llama 3.2 3B Instruct](models/llm/llama-3.2-3b-instruct) | `w4a16-g128`, `w8a8` | `w8a8` |
| VLM | [Qwen3.5 2B](models/vlm/qwen3.5-2b) | `w4a16-g128`, `w8a8` | `w8a8` |
| VLM | [Qwen3.5 4B](models/vlm/qwen3.5-4b) | `w4a16-g128`, `w8a8` | `w8a8` |

Published image names follow this pattern:

```text
ghcr.io/seeed-projects/recomputer-rk-llm/<type>/<model-id>:<platform>-<quantization>
```

To build a published model image, run [Build RKLLM model images](https://github.com/Seeed-Projects/reComputer-RK-LLM/actions/workflows/build-model-images.yml)
and select the desired scope, model, platform, and quantization.

## Runtime

The bundled RKLLM runtime is v1.3.0. The ctypes definitions in
[`app/fastapi_server_llm.py`](app/fastapi_server_llm.py) and
[`app/fastapi_server_vlm.py`](app/fastapi_server_vlm.py) match that ABI.
Runtime artifacts are kept under [`runtime/`](runtime/).

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
| `API_MODEL_NAME` | `rkllm-model` | Public API model name |
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
.github/workflows/           Environment and model image builds
docs/LLM.md                  LLM deployment guide
docs/VLM.md                  VLM deployment guide
tools/                       API performance test clients
```

## Speed test

The performance clients measure time to first token (TTFT) and time per output
token (TPOT):

```bash
python -m venv .env && source .env/bin/activate
pip install -r requirements.txt
python tools/llm_speed_test.py --help
python tools/vlm_speed_test.py --help
```

## Safety

The examples use `--privileged` and `-v /dev:/dev` because Rockchip NPU access
varies by board image. Keep the service on a trusted network; authentication
and TLS are not included.

## References

- [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm)
- [Seeed Studio reComputer](https://www.seeedstudio.com/reComputer.html)

## Community

<a href="https://github.com/Seeed-Projects/reComputer-RK-LLM/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Seeed-Projects/reComputer-RK-LLM" alt="Contributors" />
</a>

![Star History Chart](https://api.star-history.com/svg?repos=Seeed-Projects/reComputer-RK-LLM&type=Date)

[contributors-shield]: https://img.shields.io/github/contributors/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[contributors-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[forks-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/network/members
[stars-shield]: https://img.shields.io/github/stars/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[stars-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/stargazers
[issues-shield]: https://img.shields.io/github/issues/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[issues-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/issues
[license-shield]: https://img.shields.io/github/license/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[license-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/blob/main/LICENSE
