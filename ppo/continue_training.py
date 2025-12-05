import gym
import time
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback

# Get train environment configs
with open('scripts/config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

# Create a DummyVecEnv
env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "scripts:airsim-env-v0", 
        ip_address="127.0.0.1", 
        image_shape=(50,50,3),
        env_config=env_config["TrainEnv"]
    )
)])

# Wrap env as VecTransposeImage
env = VecTransposeImage(env)

# 加载预训练模型
print("加载预训练模型...")
model = PPO.load("saved_policy/ppo_navigation_policy", env=env)

# 设置设备
model.device = "cuda"

# Evaluation callback
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=4,
    best_model_save_path="./ppo",
    log_path="./ppo",
    eval_freq=500,
)

callbacks.append(eval_callback)
kwargs = {}
kwargs["callback"] = callbacks

log_name = "ppo_continue_" + str(time.time())

print("从预训练模型继续训练...")
model.learn(
    total_timesteps=100000,  # 继续训练10万步
    tb_log_name=log_name,
    reset_num_timesteps=False,  # 不重置步数计数
    **kwargs
)

# Save policy weights
model.save("ppo/ppo_continued_policy")
print("训练完成！")
