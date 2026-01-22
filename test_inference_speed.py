#!/usr/bin/env python3
"""
RKLLM Performance Test - Simplified Version
Calculates average TTFT and TPOT only
"""

import requests
import json
import time
import statistics
import sys
from typing import Dict, List, Tuple
import argparse
from datetime import datetime

class RKLLMPerformanceTester:
    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1"):
        self.base_url = base_url.rstrip('/')
        self.chat_completions_url = f"{self.base_url}/chat/completions"
    
    def _stream_response_with_metrics(self, payload: Dict) -> Tuple[Dict, str]:
        """Stream response and collect TTFT/TPOT metrics"""
        metrics = {
            'ttft_ms': 0,          # Time to first token
            'avg_tpot_ms': 0,      # Average time per output token
            'total_tokens': 0,     # Total tokens generated
            'total_time_ms': 0     # Total generation time
        }
        
        full_response = ""
        request_start_time = time.perf_counter()
        first_token_received = False
        
        try:
            # Make streaming request
            response = requests.post(
                self.chat_completions_url,
                json=payload,
                stream=True,
                timeout=120
            )
            
            if response.status_code != 200:
                print(f"Request failed: {response.status_code}")
                return metrics, ""
            
            # Process streaming response
            token_count = 0
            last_token_time = None
            token_times = []
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8', errors='ignore')
                    
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            if 'choices' not in data or len(data['choices']) == 0:
                                continue
                                
                            choice = data['choices'][0]
                            current_time = time.perf_counter()
                            
                            # Check for content
                            has_content = False
                            if 'delta' in choice and 'content' in choice['delta']:
                                content = choice['delta']['content']
                                if content and content.strip():
                                    has_content = True
                                    full_response += content
                            
                            # Record first token time
                            if not first_token_received and has_content:
                                first_token_received = True
                                metrics['ttft_ms'] = (current_time - request_start_time) * 1000
                            
                            # Calculate token intervals
                            if has_content:
                                token_count += 1
                                
                                if last_token_time is not None:
                                    token_time = (current_time - last_token_time) * 1000
                                    token_times.append(token_time)
                                
                                last_token_time = current_time
                                
                        except json.JSONDecodeError:
                            continue
            
            # Calculate final metrics
            if token_count > 0:
                metrics['total_tokens'] = token_count
                metrics['total_time_ms'] = (time.perf_counter() - request_start_time) * 1000
                
                if token_times:
                    metrics['avg_tpot_ms'] = statistics.mean(token_times)
            
            return metrics, full_response
            
        except requests.exceptions.RequestException:
            return metrics, ""
        except Exception:
            return metrics, ""
    
    def run_performance_test(self, messages: List[Dict], 
                           iterations: int = 5,
                           max_tokens: int = 100,
                           temperature: float = 0.1) -> Dict:
        """Run performance test and return average TTFT/TPOT"""
        
        print(f"\n🏁 Running Performance Test")
        print(f"   Iterations: {iterations}")
        print(f"   Max tokens per request: {max_tokens}")
        print(f"   Temperature: {temperature}")
        print(f"   Messages: {len(messages)}")
        
        # Collect metrics
        ttft_values = []
        tpot_values = []
        token_counts = []
        
        for i in range(iterations):
            print(f"\nIteration {i+1}/{iterations}...", end="")
            
            payload = {
                "model": "rkllm-model",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }
            
            metrics, _ = self._stream_response_with_metrics(payload)
            
            if metrics['total_tokens'] > 0:
                ttft_values.append(metrics['ttft_ms'])
                if metrics['avg_tpot_ms'] > 0:
                    tpot_values.append(metrics['avg_tpot_ms'])
                token_counts.append(metrics['total_tokens'])
                print(f" ✓ ({metrics['total_tokens']} tokens)")
            else:
                print(f" ✗ (failed)")
            
            # Brief pause
            if i < iterations - 1:
                time.sleep(1)
        
        # Calculate averages
        result = {
            'iterations_completed': len(ttft_values),
            'avg_ttft_ms': 0,
            'avg_tpot_ms': 0,
            'avg_tokens': 0
        }
        
        if ttft_values:
            result['avg_ttft_ms'] = statistics.mean(ttft_values)
            result['avg_ttft_s'] = result['avg_ttft_ms'] / 1000
        
        if tpot_values:
            result['avg_tpot_ms'] = statistics.mean(tpot_values)
            result['avg_speed_tps'] = 1000 / result['avg_tpot_ms'] if result['avg_tpot_ms'] > 0 else 0
        
        if token_counts:
            result['avg_tokens'] = statistics.mean(token_counts)
        
        return result

def main():
    parser = argparse.ArgumentParser(
        description="RKLLM Performance Test - Average TTFT and TPOT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_performance.py                           # Default test
  python test_performance.py --iter 10                 # 10 iterations
  python test_performance.py --tokens 200              # 200 tokens
  python test_performance.py --url http://192.168.1.100:8080/v1
        """
    )
    
    parser.add_argument('--url', type=str, default='http://127.0.0.1:8080/v1',
                       help='RKLLM server URL')
    parser.add_argument('--iter', type=int, default=5,
                       help='Number of iterations')
    parser.add_argument('--tokens', type=int, default=100,
                       help='Tokens per request')
    parser.add_argument('--prompt', type=str,
                       default="请介绍一下人工智能",
                       help='User prompt')
    parser.add_argument('--system', type=str,
                       help='System prompt (optional)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature')
    
    args = parser.parse_args()
    
    # Create tester
    tester = RKLLMPerformanceTester(args.url)
    
    # Prepare messages
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    
    # Run test
    try:
        print(f"\n{'='*50}")
        print(f"RKLLM Performance Test")
        print(f"{'='*50}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        
        results = tester.run_performance_test(
            messages=messages,
            iterations=args.iter,
            max_tokens=args.tokens,
            temperature=args.temperature
        )
        
        # Display results
        print(f"\n{'='*50}")
        print(f"RESULTS")
        print(f"{'='*50}")
        
        print(f"Completed iterations: {results['iterations_completed']}/{args.iter}")
        
        if results['avg_ttft_ms'] > 0:
            print(f"\n📊 TTFT (Time To First Token):")
            print(f"   Average: {results['avg_ttft_ms']:.2f}ms ({results['avg_ttft_s']:.4f}s)")
        
        if results['avg_tpot_ms'] > 0:
            print(f"\n📊 TPOT (Time Per Output Token):")
            print(f"   Average: {results['avg_tpot_ms']:.2f}ms")
            print(f"   Speed: {results['avg_speed_tps']:.1f} tokens/s")
        
        if results['avg_tokens'] > 0:
            print(f"\n📊 Other Metrics:")
            print(f"   Average tokens per request: {results['avg_tokens']:.1f}")
        
        print(f"\n✅ Test completed at {datetime.now().strftime('%H:%M:%S')}")
        
    except KeyboardInterrupt:
        print(f"\n\nTest interrupted")
    except Exception as e:
        print(f"\nTest failed: {e}")

if __name__ == "__main__":
    main()