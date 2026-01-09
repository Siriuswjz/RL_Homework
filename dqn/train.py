import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise


class VGG16FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1024, use_pretrained=True):
        super().__init__(observation_space, features_dim)
        
        # 使用VGG16作为骨干网络
        vgg16 = models.vgg16(pretrained=use_pretrained)
        
        # 只使用前几层卷积，保留更多空间信息
        self.features = nn.Sequential(*list(vgg16.features.children())[:23])
        
        # 部分解冻预训练权重进行微调
        if use_pretrained:
            for i, param in enumerate(self.features.parameters()):
                if i < 10:
                    param.requires_grad = False
        
        # 添加注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, observations):
        x = self.features(observations)
        attention_weights = self.attention(x)
        x = x * attention_weights
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train_dqn():
    """
    训练DQN模型
    """
    # 加载环境配置
    with open('scripts/config.yml', 'r', encoding='utf-8') as f:
        env_config = yaml.safe_load(f)

    from scripts.env import AirSimDroneEnv

    # 创建环境
    env = DummyVecEnv([lambda: Monitor(
        AirSimDroneEnv(
            ip_address="127.0.0.1", 
            image_shape=(50,50,3),
            env_config=env_config["TrainEnv"]
        )
    )])
    env = VecTransposeImage(env)

    # 策略参数
    policy_kwargs = dict(
        features_extractor_class=VGG16FeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=1024,
            use_pretrained=True  
        ),
        net_arch=[512, 512, 256],
    )

    # 检查checkpoint
    checkpoint_path = "dqn/checkpoints/dqn_390000_steps.zip"
    if os.path.exists(checkpoint_path):
        print(f"发现checkpoint: {checkpoint_path}")
        print("从checkpoint继续训练...")
        model = DQN.load(checkpoint_path, env=env, device="cuda")
        completed_steps = 390000
        remaining_steps = 110000
    else:
        print("从头开始训练...")
        model = DQN(
            'CnnPolicy',
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="cuda",
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=5000,
            batch_size=64,
            tau=0.005,
            gamma=0.995,
            train_freq=2,
            gradient_steps=2,
            target_update_interval=500,
            exploration_fraction=0.4,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            max_grad_norm=10,
            tensorboard_log="./tb_logs/",
        )
        completed_steps = 0
        remaining_steps = 500000

    # 回调函数
    eval_callback = EvalCallback(
        env,
        n_eval_episodes=5,
        best_model_save_path="./dqn/best",
        log_path="./dqn/best",
        eval_freq=10000,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./dqn/checkpoints/",
        name_prefix="dqn"
    )

    log_name = "dqn_run_" + str(time.time())

    print("开始DQN训练...")
    print(f"- VGG16 + 注意力机制")
    print(f"- 训练步数: {remaining_steps}")
    print(f"- eval_freq: 10000")

    # 训练
    model.learn(
        total_timesteps=remaining_steps,
        tb_log_name=log_name,
        reset_num_timesteps=False,
        callback=[eval_callback, checkpoint_callback]
    )

    # 保存最终模型
    model.save("dqn/final_model")
    print("DQN训练完成！")
    print(f"总训练步数: {completed_steps + remaining_steps}")


if __name__ == "__main__":
    train_dqn()
