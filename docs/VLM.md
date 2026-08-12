# VLM deployment on reComputer boards

The VLM server keeps the OpenAI-compatible API, but its inference path now
uses Rockchip's official `librknnrt.so` image encoder and `librkllmrt.so`
multimodal runtime directly. The custom `librkllm_service.so` wrapper is not
required or loaded.

The VLM uses the same v1.3.0 environment image as the LLM and selects
`MODEL_KIND=vlm` at startup. Mount a matching `.rkllm` language model and
`.rknn` vision encoder for the same target platform.

## Prerequisite

Install Docker with Buildx on the target board.

## Build and run

```bash
docker buildx build --platform linux/arm64 \
  -f docker/Dockerfile \
  -t rkllm:env --load .
```

```bash
sudo docker run --rm -it --name rkllm-vlm --privileged \
  -p 8001:8001 \
  -v /dev:/dev \
  -v ./models:/app/models:ro \
  -e MODEL_KIND=vlm \
  -e MODEL_FILE=my-vlm-language-model.rkllm \
  -e VISION_MODEL_FILE=my-vlm-vision-model.rknn \
  -e TARGET_PLATFORM=rk3576 \
  rkllm:env
```

The command stays attached to the terminal for testing. VLM interaction is
available through the HTTP API; this server does not provide terminal chat.

After startup, API documentation is available at `http://localhost:8001/docs`
and `http://localhost:8001/redoc`.

The server prints the API address during startup. For another computer on the
same network, replace `localhost` with the board's IP address.

## Use with Cherry Studio

Add a custom OpenAI-compatible provider in Cherry Studio:

- API host: `http://<board-ip>:8001/v1`
- API key: any value, for example `rkllm-local`
- Model: `rkllm-vision`

Then select the vision model and attach an image in the chat. The API accepts
the standard OpenAI multimodal message format with `text` and `image_url`
content parts. The server supports one image per request and accepts either a
public HTTP(S) image URL or a base64 data URL.

Replace the example image URL below with an image reachable from the board, or
use a local file with the speed-test tool.

## Test server

### Command line

#### Non-streaming response：

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-vision",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe the image"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/your-image.jpg"

            }
          }
        ]
      }
    ],
    "stream": false
  }'

```

#### Streaming response:

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-vision",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe the image"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/your-image.jpg"

            }
          }
        ]
      }
    ],
    "stream": true
  }'
```
### Use OpenAI API

#### Non-streaming response：

```python
import openai
import base64
import requests
import time

# Configure OpenAI client for local RKLLM Vision server
client = openai.OpenAI(
    base_url="http://localhost:8001/v1",  # Update with your server port
    api_key="dummy-key"  # Any API key works for local server
)

def test_image_description():
    """Test image description with non-streaming response"""
    print("=== Non-Streaming Image Description Test ===")
    
    # Download image from URL and convert to base64
    image_url = "https://example.com/your-image.jpg"

    
    try:
        # Download image
        print("Downloading test image...")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Convert to base64
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        print(f"Image downloaded successfully (base64 length: {len(image_base64)})")
        
        # Create request with image
        start_time = time.time()
        
        completion = client.chat.completions.create(
            model="rkllm-vision",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful AI assistant that describes images."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=100,
            top_p=1.0,
            # Use extra_body for custom parameters
            extra_body={
                "top_k": 1,
                "max_context_len": 2048,
                "rknn_core_num": 3
            },
            stream=False
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\nResponse received in {elapsed_time:.2f} seconds:")
        print(f"Request ID: {completion.id}")
        print(f"Model: {completion.model}")
        print(f"Response: {completion.choices[0].message.content}")
        print(f"Token usage: {completion.usage.total_tokens} tokens")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    print("Starting RKLLM Vision Server Tests")
    print("=" * 60)
    
    # Test 1: Basic image description
    test_image_description()
    
    print("\n" + "=" * 60)
    print("Non-streaming tests completed!")
```


#### Streaming response:


```python
import openai
import base64
import requests
import time

# Configure OpenAI client for local RKLLM Vision server
client = openai.OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy-key"
)

def test_streaming_image_description():
    """Test streaming response with image"""
    print("=== Streaming Image Description Test ===")
    
    # Download test image
    image_url = "https://example.com/your-image.jpg"

    
    try:
        print("Downloading test image...")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Convert to base64
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        print(f"Image ready (size: {len(image_base64)} bytes)")
        print("\nStarting streaming response...")
        print("Response: ", end="", flush=True)
        
        # Start timing
        start_time = time.time()
        
        # Create streaming request with extra_body
        stream = client.chat.completions.create(
            model="rkllm-vision",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail. What do you see?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=150,
            extra_body={
                "top_k": 1,
                "top_p": 1.0
            },
            stream=True  # Enable streaming
        )
        
        # Process streaming response
        full_response = ""
        token_count = 0
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
                token_count += 1
        
        # Calculate timing
        elapsed_time = time.time() - start_time
        
        print(f"\n\nStreaming completed in {elapsed_time:.2f} seconds")
        print(f"Total tokens received: {token_count}")
        print(f"Full response length: {len(full_response)} characters")
        
    except Exception as e:
        print(f"\nTest failed: {e}")

if __name__ == "__main__":
    print("RKLLM Vision Server - Streaming Tests")
    print("=" * 60)
    
    # Test basic streaming
    test_streaming_image_description()
    
    print("\n" + "=" * 60)
    print("All streaming tests completed!")
```
