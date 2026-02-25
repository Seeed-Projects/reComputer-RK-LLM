# LLM Fast start

## Step 1: Install Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh 
```

## Step 2: Install image and start server

```bash
sudo docker run -it --privileged \
  --net=host \
  --device /dev/dri \
  --device /dev/dma_heap \
  --device /dev/rknpu \
  --device /dev/mali0 \
  -v /dev:/dev \
  ghcr.io/seeed-projects/rk3576-deepseek-r1-distill-qwen:1.5b-fp16-latest
```
>Note: When you start the service, you can access `http://localhost:8001/docs` and `http://localhost:8001/redoc` to view the documentation.

## Step 3: Test Server

### Commandline
#### Non-streaming response：

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
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

#### Streaming response:

```bash
curl -N http://127.0.0.1:8001/v1/chat/completions \
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

### Use OpenAI API

#### Non-streaming response：

```python
import openai

# Configure the OpenAI client to use your local server
client = openai.OpenAI(
    base_url="http://localhost:8001/v1",  # Point to your local server
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

#### Streaming response:

```python
import openai

# Configure the OpenAI client to use your local server
client = openai.OpenAI(
    base_url="http://localhost:8001/v1",  # Point to your local server
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