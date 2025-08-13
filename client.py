'''
Author: LIANENGUANG lianenguang@gmail.com
Date: 2025-08-11 11:55:55
LastEditors: LIANENGUANG lianenguang@gmail.com
LastEditTime: 2025-08-11 12:06:41
Description: 简要描述此文件的作用
FilePath: /models/client.py
'''
import json

import requests

# 服务器地址
BASE_URL = "http://10.7.65.71:8000"

def test_root():
    """测试根路径接口"""
    response = requests.get(f"{BASE_URL}/")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def test_health():
    """测试服务健康状态"""
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def get_stats():
    """获取服务器统计信息"""
    response = requests.get(f"{BASE_URL}/stats")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def get_embeddings(texts):
    """获取文本嵌入向量"""
    payload = {
        "texts": texts,
        "max_length": 512
    }
    
    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    if response.status_code == 200:
        return response.json()["embeddings"]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def rerank_documents(query, documents):
    """对文档进行重排序"""
    payload = {
        "query": query,
        "documents": documents,
        "max_length": 512
    }
    
    response = requests.post(
        f"{BASE_URL}/rerank",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    if response.status_code == 200:
        return response.json()["scores"]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def chat_completion(messages, temperature=0.7, max_length=2048):
    """聊天对话"""
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_length": max_length,
        "do_sample": True
    }
    
    response = requests.post(
        f"{BASE_URL}/chat",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

if __name__ == "__main__":
    # 测试根路径
    print("Testing root endpoint...")
    root_result = test_root()
    if root_result:
        print("Root Response:", root_result)
    
    # 测试健康状态
    print("\nTesting health endpoint...")
    health_result = test_health()
    if health_result:
        print("Health Check:", health_result)
    
    # 测试统计信息
    print("\nTesting stats endpoint...")
    stats_result = get_stats()
    if stats_result:
        print("Stats Response:", stats_result)
    
    # 测试文本嵌入
    texts = [
        "这是一个测试句子",
        "BGE模型可以生成高质量的中文嵌入向量",
        "Docker容器运行模型服务非常方便"
    ]
    
    print("\nGetting embeddings...")
    embeddings = get_embeddings(texts)
    
    if embeddings:
        print(f"Successfully got embeddings for {len(texts)} texts")
        print(f"Embedding dimension: {len(embeddings[0])}")
        print(f"First embedding (first 5 dimensions): {embeddings[0][:5]}")
    else:
        print("Failed to get embeddings")
    
    # 测试文档重排序
    print("\nTesting reranking...")
    query = "机器学习模型训练"
    documents = [
        "深度学习是机器学习的一个分支",
        "今天天气很好，适合出去散步",
        "神经网络训练需要大量的数据",
        "我喜欢吃苹果和香蕉",
        "模型优化可以提升训练效率"
    ]
    
    scores = rerank_documents(query, documents)
    
    if scores:
        print(f"Successfully got rerank scores for {len(documents)} documents")
        # 按分数排序文档
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("Ranked documents:")
        for i, (doc, score) in enumerate(doc_scores, 1):
            print(f"{i}. [{score:.4f}] {doc}")
    else:
        print("Failed to get rerank scores")
    
    # 测试聊天对话
    print("\nTesting chat completion...")
    messages = [
        {"role": "user", "content": "什么是人工智能？请简要介绍。"}
    ]
    
    chat_result = chat_completion(messages)
    
    if chat_result:
        print("Chat Response:")
        print(f"Model: {chat_result['model']}")
        print(f"Message: {chat_result['message']}")
    else:
        print("Failed to get chat response")