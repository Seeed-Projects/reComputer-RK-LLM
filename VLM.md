# LLM Fast start

## Step 1: Install Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh 
```

## Step 2: Install image and start server

```bash
sudo docker run -it --name --privileged \
  --net=host \
  --device /dev/dri \
  --device /dev/dma_heap \
  --device /dev/rknpu \
  --device /dev/mali0 \
  -v /dev:/dev \
  ghcr.io/seeed-projects/rk3576-qwen2.5-vl:3b-w4a16-latest
```

>Note: When you start the service, you can access `http://localhost:8002/docs` and `http://localhost:8002/redoc` to view the documentation.

## Step 3: Test Server

### Commandline
#### Non-streaming response：

```bash
curl -X POST http://localhost:8002/v1/chat/completions \
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
              "url": "https://github.com/LJ-Hao/reComputer-RK-LLM/blob/main/img/test.jpeg?raw=true"

            }
          }
        ]
      }
    ],
    "stream": false,
    "max_tokens": 50
  }'

```

#### Streaming response:

```bash
curl -X POST http://localhost:8002/v1/chat/completions \
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
              "url": "https://github.com/LJ-Hao/reComputer-RK-LLM/blob/main/img/test.jpeg?raw=true"

            }
          }
        ]
      }
    ],
    "stream": true,
    "max_tokens": 50
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
    base_url="http://localhost:8002/v1",  # Update with your server port
    api_key="dummy-key"  # Any API key works for local server
)

def test_image_description():
    """Test image description with non-streaming response"""
    print("=== Non-Streaming Image Description Test ===")
    
    # Download image from URL and convert to base64
    image_url = "https://github.com/LJ-Hao/reComputer-RK-LLM/blob/main/img/test.jpeg?raw=true"

    
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
    base_url="http://localhost:8002/v1",
    api_key="dummy-key"
)

def test_streaming_image_description():
    """Test streaming response with image"""
    print("=== Streaming Image Description Test ===")
    
    # Download test image
    image_url = "https://github.com/LJ-Hao/reComputer-RK-LLM/blob/main/img/test.jpeg?raw=true"

    
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