#!/usr/bin/env python3
"""
RKLLM Vision API Streaming Speed Test Tool
Measures TTFT (Time To First Token) and TPOT (Time Per Output Token)
Always includes an image (default image provided) and uses streaming mode only.
"""

import argparse
import base64
import json
import time
import requests
import sys
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

# Default image URL (raw GitHub link to a test image)
DEFAULT_IMAGE_URL = "https://raw.githubusercontent.com/LJ-Hao/reComputer-RK-LLM/main/img/test.jpeg"

def encode_image_file(image_path: str) -> str:
    """
    Encode a local image file to a base64 data URI.
    """
    with open(image_path, "rb") as f:
        image_data = f.read()
    encoded = base64.b64encode(image_data).decode("utf-8")
    # Guess MIME type from file extension
    ext = Path(image_path).suffix.lower()
    if ext in [".png"]:
        mime = "image/png"
    elif ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext in [".gif"]:
        mime = "image/gif"
    elif ext in [".webp"]:
        mime = "image/webp"
    else:
        mime = "image/jpeg"  # fallback
    return f"data:{mime};base64,{encoded}"

def build_messages(text: str, image_source: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Construct the messages list. If no image_source is provided, use the default image URL.
    Supports local file paths, HTTP/HTTPS URLs, or data URIs.
    """
    if image_source is None:
        image_source = DEFAULT_IMAGE_URL
        print(f"[Info] Using default image URL: {image_source}")

    # Determine if image_source is a local file or a URL/data URI
    if image_source.startswith(("http://", "https://", "data:image")):
        image_url = image_source
    else:
        # Assume it's a local file path
        image_url = encode_image_file(image_source)

    content = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_url}}
    ]
    return [{"role": "user", "content": content}]

def test_streaming(url: str, model: str, messages: list,
                   temperature: float, top_p: float, max_tokens: int,
                   avg_token_chars: float = 4.0) -> Dict:
    """
    Perform a streaming request and measure TTFT, TPOT, and other metrics.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True
    }
    headers = {"Content-Type": "application/json"}

    request_start = time.perf_counter()
    first_token_time = None
    token_times = []          # timestamps of each content chunk arrival
    collected_chunks = []     # text of each content chunk

    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # remove "data: "
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get('choices', [])
                    if not choices:
                        continue
                    delta = choices[0].get('delta', {})
                    content = delta.get('content')
                    if content is not None:
                        now = time.perf_counter()
                        collected_chunks.append(content)
                        if first_token_time is None:
                            first_token_time = now
                        token_times.append(now)
    except Exception as e:
        return {"error": str(e)}

    if not token_times:
        return {"error": "No tokens received"}

    request_end = token_times[-1]  # last token arrival time

    # Calculate metrics
    ttft = first_token_time - request_start
    if len(token_times) > 1:
        intervals = np.diff(token_times)
        tpot = np.mean(intervals)
    else:
        tpot = 0.0

    total_time = request_end - request_start
    total_chunks = len(collected_chunks)
    total_chars = sum(len(chunk) for chunk in collected_chunks)
    estimated_tokens = total_chars / avg_token_chars
    chars_per_second = total_chars / total_time if total_time > 0 else 0
    estimated_tokens_per_second = estimated_tokens / total_time if total_time > 0 else 0

    return {
        "ttft": ttft,
        "tpot": tpot,
        "total_time": total_time,
        "num_chunks": total_chunks,
        "total_chars": total_chars,
        "estimated_tokens": estimated_tokens,
        "chars_per_second": chars_per_second,
        "estimated_tokens_per_second": estimated_tokens_per_second,
        "first_token_time": first_token_time,
        "last_token_time": request_end,
        "token_times": token_times
    }

def main():
    parser = argparse.ArgumentParser(
        description="Streaming speed test for RKLLM Vision API (always includes an image)."
    )
    parser.add_argument("--url", type=str, default="http://localhost:8002/v1/chat/completions",
                        help="API endpoint URL (default: http://localhost:8002/v1/chat/completions)")
    parser.add_argument("--model", type=str, default="rkllm-vision",
                        help="Model name (default: rkllm-vision)")
    parser.add_argument("--message", type=str, default='Describe this image',
                        help="User message text")
    parser.add_argument("--image", type=str, default=None,
                        help="Image file path, URL, or data URI. If not provided, a default image is used.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature parameter (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling parameter (default: 1.0)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum tokens to generate (default: 512)")
    parser.add_argument("--avg_token_chars", type=float, default=4.0,
                        help="Average characters per token for estimation (default: 4.0)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed token timings")

    args = parser.parse_args()

    # Build messages (always includes an image)
    messages = build_messages(args.message, args.image)

    print("=" * 60)
    print(f"Request URL: {args.url}")
    print(f"Model: {args.model}")
    print(f"Message: {args.message[:50]}{'...' if len(args.message)>50 else ''}")
    if args.image:
        print(f"Image: {args.image}")
    else:
        print(f"Image: default (override with --image)")
    print(f"Streaming: True")
    print(f"temperature: {args.temperature}, top_p: {args.top_p}, max_tokens: {args.max_tokens}")
    print("=" * 60)

    result = test_streaming(
        url=args.url,
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        avg_token_chars=args.avg_token_chars
    )

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        sys.exit(1)

    print("\n📊 Streaming Test Results:")
    print(f"  TTFT (Time To First Token): {result['ttft']*1000:.2f} ms")
    print(f"  TPOT (Avg. Time Per Output Token): {result['tpot']*1000:.2f} ms")
    print(f"  Total Time: {result['total_time']:.3f} s")
    print(f"  Number of content chunks: {result['num_chunks']}")
    print(f"  Total characters: {result['total_chars']}")
    print(f"  Estimated tokens (avg {args.avg_token_chars} chars/token): {result['estimated_tokens']:.1f}")
    print(f"  Character generation speed: {result['chars_per_second']:.1f} char/s")
    print(f"  Estimated token generation speed: {result['estimated_tokens_per_second']:.1f} token/s")

    if args.verbose:
        print("\n📝 Detailed token arrival times (relative to first token):")
        base_time = result['first_token_time']
        for i, t in enumerate(result['token_times']):
            print(f"    token {i+1}: {(t - base_time)*1000:.2f} ms")

if __name__ == "__main__":
    main()