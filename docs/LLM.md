# LLM deployment on reComputer boards

The LLM container uses the v1.3.0 RKLLM runtime and exposes both OpenAI- and
Ollama-compatible endpoints.

## Run a custom model

```bash
sudo docker run --rm -it \
  --name rkllm \
  --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  -v ./models:/app/models:ro \
  -e MODEL_FILE=Qwen2.5-1.5B-Instruct_RK3576_w8a8.rkllm \
  -e TARGET_PLATFORM=rk3576 \
  -e INTERACTIVE_CHAT=true \
  -e LOG_LEVEL=warning \
  rkllm:env
```

This test command stays attached to the terminal and enables terminal chat. The
server waits until the model initializes and then reports readiness at:

```text
http://localhost:8001/health
http://localhost:8001/docs
```

## OpenAI request

```bash
curl http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rkllm-model",
    "messages": [{"role": "user", "content": "Explain edge AI in one sentence."}],
    "max_tokens": 128,
    "stream": false
  }'
```

## Ollama request

```bash
curl http://localhost:8001/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"rkllm-model","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

## Build locally

```bash
docker buildx build --platform linux/arm64 -f docker/Dockerfile -t rkllm:env --load .
```

To publish a model image, download its artifacts into `models/.build/` and
build `docker/Dockerfile.model` with `BASE_IMAGE`, `MODEL_KIND`, `MODEL_FILE`,
and `TARGET_PLATFORM`. GitHub Actions automates this from the `.env` files in
`models/llm/`.
