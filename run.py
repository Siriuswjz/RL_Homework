#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN训练启动脚本
"""

import subprocess
import sys

def main():
    print("=" * 60)
    print("DQN无人机穿洞训练")
    print("=" * 60)
    print()
    print("开始训练...")
    print()
    
    # 运行训练脚本
    result = subprocess.run([sys.executable, "dqn/train.py"])
    
    if result.returncode == 0:
        print()
        print("=" * 60)
        print("✅ 训练完成！")
        print()
        print("测试模型:")
        print("  python dqn/test.py --model dqn/best/best_model.zip")
        print()
        print("查看训练曲线:")
        print("  tensorboard --logdir=./tb_logs")
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print("❌ 训练失败，请检查错误信息")
        print("=" * 60)


if __name__ == "__main__":
    main()
