# Install Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh 
```

# Start inference

## For RK3588

```bash
docker run -it --name deepseek-r1-1.5b-fp16   --privileged    --net=host    --device /dev/dri    --device /dev/dma_heap    --device /dev/rknpu    --device /dev/mali0    -v /dev:/dev      ghcr.io/lj-hao/rk3588-deepseek-r1-distill-qwen:1.5b-fp16-latest
```

## For RK3576

```bash
docker run -it --name deepseek-r1-1.5b-fp16   --privileged    --net=host    --device /dev/dri    --device /dev/dma_heap    --device /dev/rknpu    --device /dev/mali0    -v /dev:/dev      ghcr.io/lj-hao/rk3576-deepseek-r1-distill-qwen:1.5b-fp16-latest
```

>Note: When you start the service, you can access `http://localhost:8080/docs` and `http://localhost:8080/redoc` to view the documentation.
# Test API：

## Non-streaming response：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-model",
    "messages": [
      {"role": "user", "content": "Where is the capital of China？"}
    ],
    "temperature": 1,
    "max_tokens": 512,
    "top_k": 1,
    "stream": false
  }'
```

## Streaming response:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-model",
    "messages": [
      {"role": "user", "content": "Where is the capital of China？"}
    ],
    "temperature": 1,
    "max_tokens": 512,
    "top_k": 1,
    "stream": true
  }'
```

# Use OpenAI API

## Non-streaming response：

```python
import openai

# Configure the OpenAI client to use your local server
client = openai.OpenAI(
    base_url="http://localhost:8080/v1",  # Point to your local server
    api_key="dummy-key"  # The API key can be anything for this local server
)

# Test the API
response = client.chat.completions.create(
    model="rkllm-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Where is the capital of China？"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```

## Streaming response:

```python
import openai

# Configure the OpenAI client to use your local server
client = openai.OpenAI(
    base_url="http://localhost:8080/v1",  # Point to your local server
    api_key="dummy-key"  # The API key can be anything for this local server
)

# Test the API with streaming
response_stream = client.chat.completions.create(
    model="rkllm-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Where is the capital of China？"}
    ],
    temperature=0.7,
    max_tokens=512,
    stream=True  # Enable streaming
)

# Process the streaming response
for chunk in response_stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

# Speed test

> Note: A rough estimate of a model's inference speed includes both TTFT and TPOT.
> Note: You can use `python test_inference_speed.py --help` to view the help function.

```bash
python -m venv .env && source .env/bin/activate
pip install requests
python test_inference_speed.py
```

