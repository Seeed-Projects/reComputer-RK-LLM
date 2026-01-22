#!/usr/bin/env python3
"""
RKLLM Inference Performance Test Script with System Prompt Support
Tests TTFT (Time To First Token) and TPOT (Time Per Output Token)
Displays real-time streaming LLM output during tests
"""

import requests
import json
import time
import statistics
import sys
import os
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime

class RKLLMStreamingTester:
    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1"):
        """
        Initialize streaming performance tester
        
        Args:
            base_url: Base URL of RKLLM server
        """
        self.base_url = base_url.rstrip('/')
        self.chat_completions_url = f"{self.base_url}/chat/completions"
        
    def _stream_response_with_metrics(self, payload: Dict, 
                                    display_output: bool = True) -> Tuple[float, float, str, int]:
        """
        Stream response while collecting performance metrics
        
        Returns:
            (ttft_seconds, avg_tpot_seconds, full_response, total_tokens)
        """
        try:
            start_time = time.perf_counter()
            first_token_time = None
            token_times = []
            full_response = ""
            total_tokens = 0
            
            if display_output:
                print("\n🤖 LLM Response Stream:")
                print("=" * 60)
            
            # Make streaming request
            response = requests.post(
                self.chat_completions_url,
                json=payload,
                stream=True,
                timeout=120
            )
            
            if response.status_code != 200:
                print(f"❌ Request failed: {response.status_code}")
                if response.text:
                    print(f"Error details: {response.text[:200]}")
                return (float('inf'), float('inf'), "", 0)
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8', errors='ignore')
                    
                    # Skip SSE comments and empty lines
                    if line_str.startswith(':') or not line_str.strip():
                        continue
                    
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        
                        if data_str == '[DONE]':
                            if display_output:
                                print("\n" + "=" * 60)
                                print("✅ Stream completed")
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            if 'choices' in data and len(data['choices']) > 0:
                                choice = data['choices'][0]
                                
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    current_time = time.perf_counter()
                                    
                                    # Record first token time
                                    if first_token_time is None and content:
                                        first_token_time = current_time
                                        ttft = first_token_time - start_time
                                    
                                    # Record token timing
                                    if content:
                                        token_times.append(current_time)
                                        full_response += content
                                        
                                        # Estimate tokens and display
                                        token_estimate = self._estimate_tokens(content)
                                        total_tokens += token_estimate
                                        
                                        if display_output:
                                            sys.stdout.write(content)
                                            sys.stdout.flush()
                                    
                                elif 'finish_reason' in choice and choice['finish_reason']:
                                    if display_output:
                                        print(f"\n\n🏁 Finish reason: {choice['finish_reason']}")
                                    
                        except json.JSONDecodeError as e:
                            if display_output:
                                print(f"\n⚠️ JSON decode error: {e}")
                            continue
            
            end_time = time.perf_counter()
            
            # Calculate metrics
            ttft = first_token_time - start_time if first_token_time else float('inf')
            
            # Calculate TPOT from token intervals
            if len(token_times) > 1:
                intervals = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
                avg_tpot = statistics.mean(intervals) if intervals else 0
            elif total_tokens > 0:
                avg_tpot = (end_time - (first_token_time or start_time)) / total_tokens
            else:
                avg_tpot = float('inf')
            
            return (ttft, avg_tpot, full_response, total_tokens)
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return (float('inf'), float('inf'), "", 0)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return (float('inf'), float('inf'), "", 0)
    
    def test_with_system_prompt(self, system_prompt: str, user_prompt: str, 
                               max_tokens: int = 100, temperature: float = 0.1, 
                               display_output: bool = True) -> Tuple[float, float, str]:
        """
        Test with system prompt and user prompt
        
        Args:
            system_prompt: The system instruction/role setting
            user_prompt: The user's query/message
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            display_output: Whether to show streaming output
            
        Returns:
            (ttft_seconds, avg_tpot_seconds, full_response)
        """
        print(f"\n🚀 Testing with System Prompt:")
        print(f"   System: '{system_prompt[:80]}{'...' if len(system_prompt) > 80 else ''}'")
        print(f"   User: '{user_prompt[:80]}{'...' if len(user_prompt) > 80 else ''}'")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": "rkllm-model",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        ttft, avg_tpot, response, total_tokens = self._stream_response_with_metrics(
            payload, display_output
        )
        
        print(f"\n📊 Results:")
        print(f"   Time to first token: {ttft:.4f}s")
        print(f"   Avg time per token: {avg_tpot:.4f}s")
        if avg_tpot > 0 and avg_tpot != float('inf'):
            print(f"   Generation speed: {1/avg_tpot:.1f} tokens/s")
        print(f"   Total estimated tokens: {total_tokens}")
        print(f"   Response length: {len(response)} characters")
        
        return ttft, avg_tpot, response
    
    def test_ttft_with_stream(self, messages: List[Dict], max_tokens: int = 50, 
                            temperature: float = 0.1, display_output: bool = True) -> Tuple[float, str]:
        """
        Test TTFT with streaming output display
        
        Returns:
            (ttft_seconds, first_part_of_response)
        """
        print(f"\n🚀 Testing TTFT")
        if messages:
            # Display first message content preview
            first_msg = messages[0]
            role = first_msg.get('role', 'unknown')
            content = first_msg.get('content', '')
            print(f"   {role}: '{content[:100]}{'...' if len(content) > 100 else ''}'")
        
        payload = {
            "model": "rkllm-model",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        ttft, avg_tpot, response, total_tokens = self._stream_response_with_metrics(
            payload, display_output
        )
        
        print(f"\n📊 TTFT Result:")
        print(f"   Time to first token: {ttft:.4f}s")
        
        if avg_tpot != float('inf') and avg_tpot > 0:
            print(f"   Avg time per token: {avg_tpot:.4f}s")
            print(f"   Estimated speed: {1/avg_tpot:.1f} tokens/s")
        
        print(f"   Total estimated tokens: {total_tokens}")
        
        return ttft, response[:100]  # Return first 100 chars of response
    
    def test_tpot_with_stream(self, messages: List[Dict], num_tokens: int = 100,
                            temperature: float = 0.1, display_output: bool = True) -> Tuple[float, str]:
        """
        Test TPOT with streaming output display
        
        Returns:
            (avg_tpot_seconds, full_response)
        """
        print(f"\n🚀 Testing TPOT")
        if messages:
            # Display first message content preview
            first_msg = messages[0]
            role = first_msg.get('role', 'unknown')
            content = first_msg.get('content', '')
            print(f"   {role}: '{content[:100]}{'...' if len(content) > 100 else ''}'")
        print(f"   Target: {num_tokens} tokens")
        
        payload = {
            "model": "rkllm-model",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": num_tokens,
            "stream": True
        }
        
        ttft, avg_tpot, response, total_tokens = self._stream_response_with_metrics(
            payload, display_output
        )
        
        print(f"\n📊 TPOT Results:")
        if ttft != float('inf'):
            print(f"   Time to first token: {ttft:.4f}s")
        
        print(f"   Avg time per token: {avg_tpot:.4f}s")
        if avg_tpot > 0 and avg_tpot != float('inf'):
            print(f"   Generation speed: {1/avg_tpot:.1f} tokens/s")
        print(f"   Total estimated tokens: {total_tokens}")
        print(f"   Response length: {len(response)} characters")
        
        return avg_tpot, response
    
    def run_ttft_benchmark(self, messages: List[Dict], iterations: int = 5,
                          max_tokens: int = 50, temperature: float = 0.1) -> Dict:
        """
        Run TTFT benchmark with streaming output
        """
        print(f"\n{'='*70}")
        print(f"🏁 TTFT BENCHMARK (Streaming)")
        print(f"   Iterations: {iterations}")
        if messages:
            print(f"   Messages: {len(messages)} messages")
            for i, msg in enumerate(messages[:2]):  # Show first 2 messages
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                print(f"   {role}: '{content[:80]}{'...' if len(content) > 80 else ''}'")
        print(f"{'='*70}")
        
        ttft_results = []
        response_samples = []
        
        for i in range(iterations):
            print(f"\n📝 Iteration {i+1}/{iterations}")
            print(f"{'-'*40}")
            
            ttft, response_sample = self.test_ttft_with_stream(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                display_output=(i == 0)  # Show output only for first iteration
            )
            
            if ttft != float('inf'):
                ttft_results.append(ttft)
                response_samples.append(response_sample[:50])
                print(f"✅ TTFT: {ttft:.4f}s")
            else:
                print(f"❌ Iteration failed")
            
            # Brief pause between iterations
            if i < iterations - 1:
                time.sleep(2)
        
        # Calculate statistics
        if ttft_results:
            stats = {
                "iterations": len(ttft_results),
                "min_ttft": min(ttft_results),
                "max_ttft": max(ttft_results),
                "mean_ttft": statistics.mean(ttft_results),
                "median_ttft": statistics.median(ttft_results),
                "std_ttft": statistics.stdev(ttft_results) if len(ttft_results) > 1 else 0,
                "response_samples": response_samples,
                "success_rate": len(ttft_results) / iterations * 100
            }
            
            print(f"\n{'='*70}")
            print(f"📈 TTFT BENCHMARK RESULTS")
            print(f"{'='*70}")
            print(f"   Successful runs: {stats['iterations']}/{iterations} ({stats['success_rate']:.1f}%)")
            print(f"   Min TTFT: {stats['min_ttft']:.4f}s")
            print(f"   Max TTFT: {stats['max_ttft']:.4f}s")
            print(f"   Mean TTFT: {stats['mean_ttft']:.4f}s")
            print(f"   Median TTFT: {stats['median_ttft']:.4f}s")
            print(f"   Std Dev: {stats['std_ttft']:.4f}s")
            
            # Performance rating
            mean_ttft = stats['mean_ttft']
            if mean_ttft < 0.1:
                rating = "🔥 EXCELLENT"
            elif mean_ttft < 0.3:
                rating = "✅ GOOD"
            elif mean_ttft < 0.5:
                rating = "⚠️ AVERAGE"
            else:
                rating = "❌ POOR"
            
            print(f"   Performance Rating: {rating}")
            if response_samples:
                print(f"   Sample responses: {response_samples[:3]}")
            
            return stats
        else:
            print(f"\n❌ No successful TTFT tests")
            return {}
    
    def run_tpot_benchmark(self, messages: List[Dict], iterations: int = 3,
                          num_tokens: int = 100, temperature: float = 0.1) -> Dict:
        """
        Run TPOT benchmark with streaming output
        """
        print(f"\n{'='*70}")
        print(f"🏁 TPOT BENCHMARK (Streaming)")
        print(f"   Iterations: {iterations}")
        if messages:
            print(f"   Messages: {len(messages)} messages")
            for i, msg in enumerate(messages[:2]):  # Show first 2 messages
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                print(f"   {role}: '{content[:80]}{'...' if len(content) > 80 else ''}'")
        print(f"   Target tokens per iteration: {num_tokens}")
        print(f"{'='*70}")
        
        tpot_results = []
        ttft_results = []
        generation_speeds = []
        
        for i in range(iterations):
            print(f"\n📝 Iteration {i+1}/{iterations}")
            print(f"{'-'*40}")
            
            # Get both TTFT and TPOT from streaming test
            payload = {
                "model": "rkllm-model",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": num_tokens,
                "stream": True
            }
            
            ttft, avg_tpot, response, total_tokens = self._stream_response_with_metrics(
                payload, display_output=(i == 0)  # Show output only for first iteration
            )
            
            if avg_tpot != float('inf') and avg_tpot > 0:
                tpot_results.append(avg_tpot)
                if ttft != float('inf'):
                    ttft_results.append(ttft)
                
                tokens_per_second = 1 / avg_tpot
                generation_speeds.append(tokens_per_second)
                
                print(f"✅ TPOT: {avg_tpot:.4f}s/token ({tokens_per_second:.1f} tokens/s)")
                print(f"   TTFT: {ttft:.4f}s")
                print(f"   Generated: {total_tokens} estimated tokens")
            else:
                print(f"❌ Iteration failed")
            
            # Longer pause between TPOT tests
            if i < iterations - 1:
                time.sleep(3)
        
        # Calculate statistics
        if tpot_results:
            stats = {
                "iterations": len(tpot_results),
                "min_tpot": min(tpot_results),
                "max_tpot": max(tpot_results),
                "mean_tpot": statistics.mean(tpot_results),
                "median_tpot": statistics.median(tpot_results),
                "std_tpot": statistics.stdev(tpot_results) if len(tpot_results) > 1 else 0,
                "mean_ttft": statistics.mean(ttft_results) if ttft_results else 0,
                "mean_tokens_per_second": statistics.mean(generation_speeds),
                "success_rate": len(tpot_results) / iterations * 100
            }
            
            print(f"\n{'='*70}")
            print(f"📈 TPOT BENCHMARK RESULTS")
            print(f"{'='*70}")
            print(f"   Successful runs: {stats['iterations']}/{iterations} ({stats['success_rate']:.1f}%)")
            print(f"   Min TPOT: {stats['min_tpot']:.4f}s/token")
            print(f"   Max TPOT: {stats['max_tpot']:.4f}s/token")
            print(f"   Mean TPOT: {stats['mean_tpot']:.4f}s/token")
            print(f"   Median TPOT: {stats['median_tpot']:.4f}s/token")
            print(f"   Mean TTFT: {stats['mean_ttft']:.4f}s")
            print(f"   Generation speed: {stats['mean_tokens_per_second']:.1f} tokens/s")
            
            # Performance rating
            speed = stats['mean_tokens_per_second']
            if speed > 50:
                rating = "🔥 EXCELLENT"
            elif speed > 20:
                rating = "✅ GOOD"
            elif speed > 5:
                rating = "⚠️ AVERAGE"
            else:
                rating = "❌ POOR"
            
            print(f"   Performance Rating: {rating}")
            
            return stats
        else:
            print(f"\n❌ No successful TPOT tests")
            return {}
    
    def run_system_prompt_test(self, system_prompt: str, user_prompt: str = None,
                              iterations: int = 3, num_tokens: int = 100,
                              temperature: float = 0.1):
        """
        Run comprehensive test with system prompt
        
        Args:
            system_prompt: The system instruction
            user_prompt: User query (if None, uses default)
            iterations: Number of test iterations
            num_tokens: Maximum tokens to generate
            temperature: Temperature for generation
        """
        if user_prompt is None:
            user_prompt = "请详细介绍相对论"
        
        print(f"\n{'='*70}")
        print(f"🚀 RKLLM SYSTEM PROMPT PERFORMANCE TEST")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Server: {self.base_url}")
        print(f"{'='*70}")
        
        # Create messages with system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Check server health
        print("\n🔍 Checking server connection...")
        try:
            health_url = self.base_url.replace('/v1', '/health')
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Server is healthy")
                print(f"   Model initialized: {health_data.get('model_initialized', 'Unknown')}")
                print(f"   Active requests: {health_data.get('active_requests', 0)}")
            else:
                print(f"⚠️ Server returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️ Could not check server health: {e}")
        
        # Run TTFT benchmark
        print(f"\n{'='*70}")
        print(f"1. TTFT BENCHMARK WITH SYSTEM PROMPT")
        ttft_stats = self.run_ttft_benchmark(
            messages=messages,
            iterations=iterations,
            max_tokens=min(50, num_tokens),
            temperature=temperature
        )
        
        # Run TPOT benchmark
        print(f"\n{'='*70}")
        print(f"2. TPOT BENCHMARK WITH SYSTEM PROMPT")
        tpot_stats = self.run_tpot_benchmark(
            messages=messages,
            iterations=min(iterations, 3),  # TPOT tests are slower
            num_tokens=num_tokens,
            temperature=temperature
        )
        
        # Summary report
        print(f"\n{'='*70}")
        print(f"🎯 SYSTEM PROMPT PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n📋 Test Configuration:")
        print(f"   System prompt: '{system_prompt[:100]}{'...' if len(system_prompt) > 100 else ''}'")
        print(f"   User prompt: '{user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}'")
        print(f"   Temperature: {temperature}")
        
        if ttft_stats:
            print(f"\n📊 TTFT Performance:")
            print(f"   Mean TTFT: {ttft_stats['mean_ttft']:.4f}s")
            print(f"   Stability: ±{ttft_stats['std_ttft']:.4f}s")
            print(f"   Success rate: {ttft_stats['success_rate']:.1f}%")
        
        if tpot_stats:
            print(f"\n📊 TPOT Performance:")
            print(f"   Mean TPOT: {tpot_stats['mean_tpot']:.4f}s/token")
            print(f"   Generation speed: {tpot_stats['mean_tokens_per_second']:.1f} tokens/s")
            print(f"   Mean TTFT: {tpot_stats['mean_ttft']:.4f}s")
            print(f"   Success rate: {tpot_stats['success_rate']:.1f}%")
        
        # Single streaming test to show behavior
        print(f"\n{'='*70}")
        print(f"3. BEHAVIOR DEMONSTRATION")
        print(f"{'='*70}")
        
        print(f"\n💡 Testing response behavior with system prompt...")
        ttft, tpot, response = self.test_with_system_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=num_tokens,
            temperature=temperature,
            display_output=True
        )
        
        print(f"\n✅ System prompt test completed at {datetime.now().strftime('%H:%M:%S')}")
        
        return {
            "ttft_stats": ttft_stats,
            "tpot_stats": tpot_stats,
            "sample_response": response[:200] if response else ""
        }
    
    def run_comprehensive_test(self, system_prompt: str = None, user_prompt: str = None,
                             ttft_iterations: int = 3, tpot_iterations: int = 2,
                             num_tokens: int = 100, temperature: float = 0.1):
        """
        Run comprehensive test with optional system prompt
        
        Args:
            system_prompt: Optional system instruction
            user_prompt: User query
            ttft_iterations: Number of TTFT iterations
            tpot_iterations: Number of TPOT iterations
            num_tokens: Maximum tokens to generate
            temperature: Temperature for generation
        """
        if user_prompt is None:
            user_prompt = "Explain the fundamental concepts of machine learning and artificial intelligence."
        
        print(f"\n{'='*70}")
        print(f"🚀 RKLLM COMPREHENSIVE PERFORMANCE TEST")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Server: {self.base_url}")
        if system_prompt:
            print(f"   With system prompt: Yes")
        print(f"{'='*70}")
        
        # Create messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        # Check server health
        print("\n🔍 Checking server connection...")
        try:
            health_url = self.base_url.replace('/v1', '/health')
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Server is healthy")
                print(f"   Model initialized: {health_data.get('model_initialized', 'Unknown')}")
                print(f"   Active requests: {health_data.get('active_requests', 0)}")
            else:
                print(f"⚠️ Server returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️ Could not check server health: {e}")
        
        # Run TTFT benchmark
        print(f"\n{'='*70}")
        print(f"1. TTFT BENCHMARK")
        ttft_stats = self.run_ttft_benchmark(
            messages=messages,
            iterations=ttft_iterations,
            max_tokens=min(50, num_tokens),
            temperature=temperature
        )
        
        # Run TPOT benchmark
        print(f"\n{'='*70}")
        print(f"2. TPOT BENCHMARK")
        tpot_stats = self.run_tpot_benchmark(
            messages=messages,
            iterations=tpot_iterations,
            num_tokens=num_tokens,
            temperature=temperature
        )
        
        # Summary report
        print(f"\n{'='*70}")
        print(f"🎯 PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        
        if ttft_stats:
            print(f"\n📊 TTFT Performance:")
            print(f"   Mean TTFT: {ttft_stats['mean_ttft']:.4f}s")
            print(f"   Stability: ±{ttft_stats['std_ttft']:.4f}s")
            print(f"   Success rate: {ttft_stats['success_rate']:.1f}%")
        
        if tpot_stats:
            print(f"\n📊 TPOT Performance:")
            print(f"   Mean TPOT: {tpot_stats['mean_tpot']:.4f}s/token")
            print(f"   Generation speed: {tpot_stats['mean_tokens_per_second']:.1f} tokens/s")
            print(f"   Mean TTFT: {tpot_stats['mean_ttft']:.4f}s")
            print(f"   Success rate: {tpot_stats['success_rate']:.1f}%")
        
        print(f"\n✅ Test completed at {datetime.now().strftime('%H:%M:%S')}")
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count in text"""
        # Simple estimation for mixed language text
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.3)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="RKLLM Inference Performance Test with System Prompt Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_inference_stream.py                             # Comprehensive test
  python test_inference_stream.py --test ttft --iter 5       # TTFT test only
  python test_inference_stream.py --test tpot --tokens 200   # TPOT test only
  python test_inference_stream.py --system "You are a helpful assistant" --prompt "Explain AI"
  python test_inference_stream.py --system-file system.txt   # Load system prompt from file
  python test_inference_stream.py --url http://192.168.1.100:8080/v1
        """
    )
    
    parser.add_argument('--url', type=str, default='http://127.0.0.1:8080/v1',
                       help='RKLLM server URL (default: http://127.0.0.1:8080/v1)')
    parser.add_argument('--test', type=str, choices=['ttft', 'tpot', 'all', 'system'],
                       default='all', help='Test type (default: all)')
    parser.add_argument('--iter', type=int, default=3,
                       help='Number of iterations per test (default: 3)')
    parser.add_argument('--tokens', type=int, default=100,
                       help='Number of tokens to generate (default: 100)')
    parser.add_argument('--prompt', type=str,
                       default="请详细介绍相对论",
                       help='User prompt (default: 请详细介绍相对论)')
    parser.add_argument('--system', type=str,
                       help='System prompt/instruction')
    parser.add_argument('--system-file', type=str,
                       help='File containing system prompt')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for generation (default: 0.1)')
    parser.add_argument('--no-stream', action='store_true',
                       help='Disable streaming output display')
    
    args = parser.parse_args()
    
    # Load system prompt from file if specified
    system_prompt = args.system
    if args.system_file:
        try:
            with open(args.system_file, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
            print(f"📄 Loaded system prompt from {args.system_file}")
            print(f"   Length: {len(system_prompt)} characters")
        except Exception as e:
            print(f"❌ Failed to load system prompt file: {e}")
            return
    
    # Create tester
    tester = RKLLMStreamingTester(args.url)
    
    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": args.prompt})
    
    # Run tests
    try:
        if args.test == 'system' and system_prompt:
            print(f"\n🔧 Configuration:")
            print(f"   Server: {args.url}")
            print(f"   System prompt length: {len(system_prompt)} characters")
            print(f"   User prompt length: {len(args.prompt)} characters")
            print(f"   Temperature: {args.temperature}")
            print(f"   Max tokens: {args.tokens}")
            
            tester.run_system_prompt_test(
                system_prompt=system_prompt,
                user_prompt=args.prompt,
                iterations=args.iter,
                num_tokens=args.tokens,
                temperature=args.temperature
            )
        
        elif args.test in ['ttft', 'all']:
            print(f"\n🔧 Configuration:")
            print(f"   Server: {args.url}")
            if system_prompt:
                print(f"   With system prompt: Yes ({len(system_prompt)} chars)")
            print(f"   User prompt length: {len(args.prompt)} characters")
            print(f"   Temperature: {args.temperature}")
            print(f"   Max tokens: {args.tokens}")
            
            if args.test == 'ttft' or args.test == 'all':
                tester.run_ttft_benchmark(
                    messages=messages,
                    iterations=args.iter,
                    max_tokens=min(50, args.tokens),
                    temperature=args.temperature
                )
        
        if args.test in ['tpot', 'all'] and args.test != 'ttft':
            if args.test == 'tpot':
                print(f"\n🔧 Configuration:")
                print(f"   Server: {args.url}")
                if system_prompt:
                    print(f"   With system prompt: Yes ({len(system_prompt)} chars)")
                print(f"   User prompt length: {len(args.prompt)} characters")
                print(f"   Temperature: {args.temperature}")
                print(f"   Target tokens: {args.tokens}")
            
            tester.run_tpot_benchmark(
                messages=messages,
                iterations=min(args.iter, 3),  # TPOT tests are slower
                num_tokens=args.tokens,
                temperature=args.temperature
            )
        
        if args.test == 'all':
            tester.run_comprehensive_test(
                system_prompt=system_prompt,
                user_prompt=args.prompt,
                ttft_iterations=args.iter,
                tpot_iterations=min(args.iter, 2),
                num_tokens=args.tokens,
                temperature=args.temperature
            )
        
        print(f"\n🎉 All tests completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()