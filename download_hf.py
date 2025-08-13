'''
Author: LIANENGUANG lianenguang@gmail.com
Date: 2025-08-11 18:53:37
LastEditors: LIANENGUANG lianenguang@gmail.com
LastEditTime: 2025-08-13 08:44:57
Description: 简要描述此文件的作用
FilePath: /models/download_hf.py
'''
#!/usr/bin/env python3

import os
import subprocess

# 设置镜像源环境变量
env = os.environ.copy()
env['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("使用hf命令下载模型...")

# 使用正确的hf download命令参数
cmd = [
    'hf', 'download',
    'Qwen/Qwen3-30B-A3B-Instruct-2507',
    '--local-dir', '/Users/lianenguang/models/data/Qwen3-30B-A3B-Instruct-2507'
]

try:
    subprocess.run(cmd, env=env, check=True)
    print("✅ 下载完成！")
except subprocess.CalledProcessError as e:
    print(f"❌ 下载失败: {e}")
except FileNotFoundError:
    print("❌ 找不到hf命令，请安装最新版huggingface_hub")
    print("pip install --upgrade huggingface_hub")