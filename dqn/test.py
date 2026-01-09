import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from scripts.env import ContinuousTestEnv


def test_model(model_path, num_steps=5000):
    print(f"加载模型: {model_path}")
    
    with open('scripts/config.yml', 'r', encoding='utf-8') as f:
        env_config = yaml.safe_load(f)
    
    print("创建测试环境...")
    env = DummyVecEnv([lambda: ContinuousTestEnv(
        ip_address="127.0.0.1",
        image_shape=(50, 50, 3),
        env_config=env_config["TrainEnv"]
    )])
    env = VecTransposeImage(env)
    
    model = DQN.load(model_path, env=env, device="cuda")
    print("模型加载成功！")
    print("=" * 60)
    print("开始测试，按 Ctrl+C 停止")
    print("=" * 60)
    
    obs = env.reset()
    
    try:
        for step in range(num_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(action)
            
            if step % 500 == 0 and step > 0:
                print(f"已执行 {step} 步...")
    except KeyboardInterrupt:
        print("\n测试被中断")
    
    test_env = env.envs[0]
    print("\n" + "=" * 60)
    print("测试结果:")
    print(f"  总episodes: {test_env.eps_n}")
    if len(test_env.agent_traveled) > 0:
        print(f"  平均飞行距离: {np.mean(test_env.agent_traveled):.2f} m")
        print(f"  最大飞行距离: {np.max(test_env.agent_traveled):.2f} m")
        print(f"  平均穿洞数: {int(np.mean(test_env.agent_traveled)//4)}")
        print(f"  最大穿洞数: {int(np.max(test_env.agent_traveled)//4)}")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试DQN模型")
    parser.add_argument("--model", type=str, 
                       default="dqn/enhanced/best_model.zip",
                       help="模型文件路径")
    parser.add_argument("--steps", type=int, default=5000,
                       help="测试步数")
    
    args = parser.parse_args()
    
    print("DQN模型测试")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"测试步数: {args.steps}")
    print("=" * 60)
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        sys.exit(1)
    
    test_model(args.model, args.steps)
