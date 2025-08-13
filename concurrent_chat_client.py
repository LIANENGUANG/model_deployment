'''
Author: LIANENGUANG lianenguang@gmail.com
Date: 2025-08-13
Description: 并发聊天客户端性能测试
FilePath: /models/concurrent_chat_client.py
'''
import asyncio
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import aiohttp

# 服务器地址
BASE_URL = "http://10.7.65.71:8000"

class ConcurrentChatClient:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        
    async def single_chat_request(self, session: aiohttp.ClientSession, 
                                  messages: List[Dict], 
                                  request_id: int = 0) -> Tuple[float, Dict]:
        """发送单个聊天请求并记录时间"""
        payload = {
            "messages": messages
        }
        
        start_time = time.time()
        try:
            async with session.post(
                f"{self.base_url}/chat",
                headers={"Content-Type": "application/json"},
                json=payload
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return response_time, {"status": "success", "data": result, "request_id": request_id}
                else:
                    error_text = await response.text()
                    return response_time, {
                        "status": "error", 
                        "error": f"HTTP {response.status}: {error_text}",
                        "request_id": request_id
                    }
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            return response_time, {"status": "error", "error": str(e), "request_id": request_id}
    
    async def concurrent_chat_test(self, messages_list: List[List[Dict]], 
                                   concurrency: int = 10) -> List[Tuple[float, Dict]]:
        """并发发送多个聊天请求"""
        connector = aiohttp.TCPConnector(limit=concurrency * 2, limit_per_host=concurrency * 2)
        timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            for i, messages in enumerate(messages_list):
                task = self.single_chat_request(session, messages, i)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append((0.0, {
                        "status": "error", 
                        "error": f"Task exception: {str(result)}",
                        "request_id": i
                    }))
                else:
                    processed_results.append(result)
            
            return processed_results
    
    def generate_test_messages(self, num_requests: int) -> List[List[Dict]]:
        """生成测试用的消息列表"""
        test_questions = [
            "什么是人工智能？请简要介绍。",
            "请解释一下机器学习和深度学习的区别。",
            "如何优化神经网络的训练速度？",
            "Python中有哪些常用的机器学习库？",
            "什么是Transformer架构？",
            "请介绍一下GPT模型的工作原理。",
            "如何评估一个机器学习模型的性能？",
            "什么是过拟合？如何避免？",
            "请解释一下梯度下降算法。",
            "什么是卷积神经网络？它有什么优势？"
        ]
        
        messages_list = []
        for i in range(num_requests):
            question = test_questions[i % len(test_questions)]
            messages = [{"role": "user", "content": f"{question} (请求#{i+1})"}]
            messages_list.append(messages)
        
        return messages_list
    
    def analyze_results(self, results: List[Tuple[float, Dict]]) -> Dict:
        """分析测试结果"""
        response_times = []
        successful_requests = 0
        failed_requests = 0
        errors = []
        
        for response_time, result in results:
            if result["status"] == "success":
                successful_requests += 1
                response_times.append(response_time)
            else:
                failed_requests += 1
                errors.append(result["error"])
        
        if response_times:
            stats = {
                "total_requests": len(results),
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": successful_requests / len(results) * 100,
                "response_times": {
                    "min": min(response_times),
                    "max": max(response_times),
                    "mean": statistics.mean(response_times),
                    "median": statistics.median(response_times),
                    "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
                },
                "errors": errors[:5]  # 只显示前5个错误
            }
        else:
            stats = {
                "total_requests": len(results),
                "successful_requests": 0,
                "failed_requests": failed_requests,
                "success_rate": 0,
                "response_times": None,
                "errors": errors[:5]
            }
        
        return stats
    
    def print_results(self, stats: Dict, test_name: str):
        """打印测试结果"""
        print(f"\n{'='*60}")
        print(f"测试结果: {test_name}")
        print(f"{'='*60}")
        print(f"总请求数: {stats['total_requests']}")
        print(f"成功请求: {stats['successful_requests']}")
        print(f"失败请求: {stats['failed_requests']}")
        print(f"成功率: {stats['success_rate']:.2f}%")
        
        if stats['response_times']:
            rt = stats['response_times']
            print("\n响应时间统计 (秒):")
            print(f"  最小值: {rt['min']:.3f}s")
            print(f"  最大值: {rt['max']:.3f}s")
            print(f"  平均值: {rt['mean']:.3f}s")
            print(f"  中位数: {rt['median']:.3f}s")
            print(f"  标准差: {rt['std_dev']:.3f}s")
            
            # 计算QPS
            if rt['mean'] > 0:
                print(f"  平均QPS: {1/rt['mean']:.2f} 请求/秒")
        
        if stats['errors']:
            print("\n前几个错误:")
            for i, error in enumerate(stats['errors'], 1):
                print(f"  {i}. {error}")
    
    async def run_performance_test(self):
        """运行性能测试"""
        print("开始并发聊天性能测试...")
        print(f"目标服务器: {self.base_url}")
        
        # 测试不同的并发数量
        test_configs = [
            {"num_requests": 5, "concurrency": 1, "name": "单线程基准测试"},
            {"num_requests": 10, "concurrency": 5, "name": "低并发测试 (5并发)"},
            {"num_requests": 20, "concurrency": 10, "name": "中等并发测试 (10并发)"},
            {"num_requests": 30, "concurrency": 15, "name": "高并发测试 (15并发)"},
        ]
        
        all_results = {}
        
        for config in test_configs:
            print(f"\n正在运行: {config['name']}")
            print(f"请求数: {config['num_requests']}, 并发数: {config['concurrency']}")
            
            messages_list = self.generate_test_messages(config['num_requests'])
            
            start_time = time.time()
            results = await self.concurrent_chat_test(messages_list, config['concurrency'])
            total_time = time.time() - start_time
            
            stats = self.analyze_results(results)
            stats['total_time'] = total_time
            stats['overall_qps'] = config['num_requests'] / total_time
            
            all_results[config['name']] = stats
            
            self.print_results(stats, config['name'])
            print(f"总耗时: {total_time:.3f}s")
            print(f"整体QPS: {stats['overall_qps']:.2f} 请求/秒")
            
            # 显示一个成功的响应示例
            success_examples = [r for r in results if r[1]["status"] == "success"]
            if success_examples:
                example = success_examples[0][1]["data"]
                print(f"\n响应示例 (请求#{example.get('request_id', 0)}):")
                print(f"  模型: {example.get('model', 'N/A')}")
                message = example.get('message', '')
                print(f"  回复: {message[:100]}{'...' if len(message) > 100 else ''}")
            
            # 等待一下再进行下一个测试
            await asyncio.sleep(2)
        
        # 打印汇总结果
        print(f"\n{'='*80}")
        print("测试汇总")
        print(f"{'='*80}")
        print(f"{'测试名称':<25} {'成功率':<8} {'平均响应时间':<12} {'平均QPS':<10} {'整体QPS':<10}")
        print(f"{'-'*80}")
        
        for name, stats in all_results.items():
            success_rate = f"{stats['success_rate']:.1f}%"
            avg_response = f"{stats['response_times']['mean']:.3f}s" if stats['response_times'] else "N/A"
            avg_qps = f"{1/stats['response_times']['mean']:.2f}" if stats['response_times'] and stats['response_times']['mean'] > 0 else "N/A"
            overall_qps = f"{stats['overall_qps']:.2f}"
            
            print(f"{name:<25} {success_rate:<8} {avg_response:<12} {avg_qps:<10} {overall_qps:<10}")

def main():
    """主函数"""
    client = ConcurrentChatClient()
    
    # 首先测试连接
    print("测试服务器连接...")
    try:
        import requests
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✓ 服务器连接正常")
        else:
            print(f"✗ 服务器响应异常: {response.status_code}")
            return
    except Exception as e:
        print(f"✗ 无法连接到服务器: {e}")
        return
    
    # 运行异步性能测试
    asyncio.run(client.run_performance_test())

if __name__ == "__main__":
    main()
