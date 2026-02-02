[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![Huggingface][huggingface-shield]][huggingface-url]

# Install Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh 
```

# Start inference

### LLM

| Device | Model |
|--------|-------|
| **RK3588** | [rk3588-deepseek-r1-distill-qwen:7b-w8a8-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3588-deepseek-r1-distill-qwen/662247747?tag=7b-w8a8-latest)<br>[rk3588-deepseek-r1-distill-qwen:1.5b-fp16-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3588-deepseek-r1-distill-qwen/662243577?tag=1.5b-fp16-latest)<br>[rk3588-deepseek-r1-distill-qwen:1.5b-w8a8-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3588-deepseek-r1-distill-qwen/662236226?tag=1.5b-w8a8-latest) | 
| **RK3576** | [rk3576-deepseek-r1-distill-qwen:7b-w4a16-g128-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3576-deepseek-r1-distill-qwen/662240247?tag=7b-w4a16-g128-latest)<br>[rk3576-deepseek-r1-distill-qwen:7b-w4a16-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3576-deepseek-r1-distill-qwen/662239597?tag=7b-w4a16-latest)<br>[rk3576-deepseek-r1-distill-qwen:1.5b-fp16-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3576-deepseek-r1-distill-qwen/662236690?tag=1.5b-fp16-latest)<br>[rk3576-deepseek-r1-distill-qwen:1.5b-w4a16-g128-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3576-deepseek-r1-distill-qwen/662235949?tag=1.5b-w4a16-g128-latest)<br>[rk3576-deepseek-r1-distill-qwen:1.5b-w4a16-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3576-deepseek-r1-distill-qwen/662234478?tag=1.5b-w4a16-latest) | 

For example:

```bash
docker run -it --name deepseek-r1-1.5b-fp16 \
  --privileged \
  --net=host \
  --device /dev/dri \
  --device /dev/dma_heap \
  --device /dev/rknpu \
  --device /dev/mali0 \
  -v /dev:/dev \
  ghcr.io/lj-hao/rk3588-deepseek-r1-distill-qwen:1.5b-fp16-latest
```

>Note: When you start the service, you can access `http://localhost:8001/docs` and `http://localhost:8001/redoc` to view the documentation.

### VLM

| Device | Model |
|--------|-------|
| **RK3588** | [rk3588-qwen2-vl:7b-w8a8-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3588-qwen2-vl/666595093?tag=7b-w8a8-latest)<br>[rk3588-qwen2-vl:2b-w8a8-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3588-qwen2-vl/666591327?tag=2b-w8a8-latest)<br> | 
| **RK3576** | [rk3576-qwen2.5-vl:3b-w4a16-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3576-qwen2.5-vl)<br>| 


For example:
```bash
sudo docker run -it --name qwen2.5-3b-w4a16-vl \
  --privileged \
  --net=host \
  --device /dev/dri \
  --device /dev/dma_heap \
  --device /dev/rknpu \
  --device /dev/mali0 \
  -v /dev:/dev \
  ghcr.io/lj-hao/rk3576-qwen2.5-vl:3b-w4a16-latest
```

>Note: When you start the service, you can access `http://localhost:8002/docs` and `http://localhost:8002/redoc` to view the documentation.

# API Test
## LLM
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

## VLM
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
              "url": "https://github.com/LJ-Hao/reComputer-RK-LLM/blob/main/img/test.jpeg"
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
              "url": "https://github.com/LJ-Hao/reComputer-RK-LLM/blob/main/img/test.jpeg"
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
    image_url = "https://github.com/LJ-Hao/reComputer-RK-LLM/raw/main/img/test.jpeg"
    
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
            top_k=1,
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

def test_multiple_images():
    """Test with multiple images in the same request"""
    print("\n=== Multiple Images Test ===")
    
    # We'll use the same image twice for demonstration
    image_url = "https://github.com/LJ-Hao/reComputer-RK-LLM/raw/main/img/test.jpeg"
    
    try:
        # Download image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        
        print("Testing with two images in the same request...")
        
        completion = client.chat.completions.create(
            model="rkllm-vision",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Compare these two images. Are they similar or different?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "This is the same image again for comparison."
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
            max_tokens=120,
            temperature=0.5
        )
        
        print(f"Response: {completion.choices[0].message.content}")
        
    except Exception as e:
        print(f"Test failed: {e}")

def test_image_question_answer():
    """Test question-answering about an image"""
    print("\n=== Image Q&A Test ===")
    
    image_url = "https://github.com/LJ-Hao/reComputer-RK-LLM/raw/main/img/test.jpeg"
    
    try:
        # Download image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        
        print("Asking specific questions about the image...")
        
        questions = [
            "What objects can you see in this image?",
            "What colors are prominent in this image?",
            "What might be happening in this scene?",
            "If this is a product, what is its main function?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}")
            
            completion = client.chat.completions.create(
                model="rkllm-vision",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": question
                            }
                        ]
                    }
                ],
                max_tokens=80,
                temperature=0.7
            )
            
            print(f"Answer: {completion.choices[0].message.content}")
            
    except Exception as e:
        print(f"Test failed: {e}")

def test_direct_image_url():
    """Test using direct image URL (server downloads the image)"""
    print("\n=== Direct Image URL Test ===")
    
    try:
        completion = client.chat.completions.create(
            model="rkllm-vision",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this image and tell me what you see."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://github.com/LJ-Hao/reComputer-RK-LLM/raw/main/img/test.jpeg"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        print(f"Response: {completion.choices[0].message.content}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    print("Starting RKLLM Vision Server Tests")
    print("=" * 60)
    
    # Test 1: Basic image description
    test_image_description()
    
    # Test 2: Multiple images
    test_multiple_images()
    
    # Test 3: Image Q&A
    test_image_question_answer()
    
    # Test 4: Direct URL (if server supports it)
    test_direct_image_url()
    
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
    image_url = "https://github.com/LJ-Hao/reComputer-RK-LLM/raw/main/img/test.jpeg"
    
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
        
        # Create streaming request
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

def test_streaming_with_system_prompt():
    """Test streaming with system prompt and image"""
    print("\n=== Streaming with System Prompt ===")
    
    image_url = "https://github.com/LJ-Hao/reComputer-RK-LLM/raw/main/img/test.jpeg"
    
    try:
        # Download and encode image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        
        print("Analyzing image as an art critic...")
        print("Art Critic: ", end="", flush=True)
        
        stream = client.chat.completions.create(
            model="rkllm-vision",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert art critic. Analyze images from artistic perspective."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Please analyze this image as an art critic. Discuss composition, colors, and artistic elements."
                        }
                    ]
                }
            ],
            temperature=0.8,
            max_tokens=120,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print(f"\nAnalysis complete ({len(full_response)} characters)")
        
    except Exception as e:
        print(f"\nTest failed: {e}")

def test_interactive_streaming():
    """Test interactive streaming with follow-up questions"""
    print("\n=== Interactive Streaming Test ===")
    
    image_url = "https://github.com/LJ-Hao/reComputer-RK-LLM/raw/main/img/test.jpeg"
    
    try:
        # Download image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        
        # First question
        print("Q1: What is the main subject of this image?")
        print("A1: ", end="", flush=True)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "What is the main subject of this image?"
                    }
                ]
            }
        ]
        
        stream1 = client.chat.completions.create(
            model="rkllm-vision",
            messages=messages,
            max_tokens=60,
            stream=True
        )
        
        response1 = ""
        for chunk in stream1:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                response1 += content
        
        # Add first response to conversation history
        messages.append({"role": "assistant", "content": response1})
        
        # Second question based on first response
        print("\n\nQ2: Can you provide more technical details about the image?")
        print("A2: ", end="", flush=True)
        
        messages.append({
            "role": "user",
            "content": "Can you provide more technical details about the image?"
        })
        
        stream2 = client.chat.completions.create(
            model="rkllm-vision",
            messages=messages,
            max_tokens=80,
            stream=True
        )
        
        for chunk in stream2:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
        
        print("\n\nInteractive conversation completed.")
        
    except Exception as e:
        print(f"\nTest failed: {e}")

def test_streaming_with_different_temperatures():
    """Test streaming with different temperature settings"""
    print("\n=== Temperature Comparison Test ===")
    
    image_url = "https://github.com/LJ-Hao/reComputer-RK-LLM/raw/main/img/test.jpeg"
    
    try:
        # Download image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        
        temperatures = [0.1, 0.5, 0.9, 1.2]
        
        for temp in temperatures:
            print(f"\nTemperature: {temp}")
            print("Response: ", end="", flush=True)
            
            stream = client.chat.completions.create(
                model="rkllm-vision",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Briefly describe this image."
                            }
                        ]
                    }
                ],
                temperature=temp,
                max_tokens=50,
                stream=True
            )
            
            response_text = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    response_text += content
            
            print(f"\nResponse length: {len(response_text)} chars")
            print("-" * 40)
            
    except Exception as e:
        print(f"\nTest failed: {e}")

if __name__ == "__main__":
    print("RKLLM Vision Server - Streaming Tests")
    print("=" * 60)
    
    # Test basic streaming
    test_streaming_image_description()
    
    # Test with system prompt
    test_streaming_with_system_prompt()
    
    # Test interactive conversation
    test_interactive_streaming()
    
    # Test temperature variations
    test_streaming_with_different_temperatures()
    
    print("\n" + "=" * 60)
    print("All streaming tests completed!")
```
# Speed test

> Note: A rough estimate of a model's inference speed includes both TTFT and TPOT.
> Note: You can use `python test_inference_speed.py --help` to view the help function.

```bash
python -m venv .env && source .env/bin/activate
pip install requests
python test_inference_speed.py
```

# 💞 Top contributors:

<a href="https://github.com/Seeed-Projects/reComputer-RK-LLM/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Seeed-Projects/reComputer-RK-LLM" alt="contrib.rocks image" />
</a>

# 🌟 Star History

![Star History Chart](https://api.star-history.com/svg?repos=Seeed-Projects/reComputer-RK-LLM&type=Date)

Reference: [rknn-llm](https://github.com/airockchip/rknn-llm/tree/main)


[contributors-shield]: https://img.shields.io/github/contributors/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[contributors-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[forks-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/network/members
[stars-shield]: https://img.shields.io/github/stars/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[stars-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/stargazers
[issues-shield]: https://img.shields.io/github/issues/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[issues-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/issues
[license-shield]: https://img.shields.io/github/license/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[license-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/blob/master/LICENSE.txt
[huggingface-shield]: https://img.shields.io/github/license/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[huggingface-url]: https://huggingface.co/JiahaoLi