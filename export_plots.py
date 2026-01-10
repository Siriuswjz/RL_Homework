import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_tensorboard_data(log_dir, tag):
    """从TensorBoard日志中加载数据"""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    try:
        data = ea.Scalars(tag)
        steps = [x.step for x in data]
        values = [x.value for x in data]
        return steps, values
    except:
        return None, None

def merge_logs(log_dirs, tag):
    """合并多个日志的数据"""
    all_steps = []
    all_values = []
    
    for log_dir in sorted(log_dirs):
        steps, values = load_tensorboard_data(log_dir, tag)
        if steps and values:
            all_steps.extend(steps)
            all_values.extend(values)
    
    return all_steps, all_values

def plot_training_curves():
    """绘制训练曲线"""
    
    # 合并两个日志：第一次训练0-390K，第二次训练390K-500K
    enhanced_logs = [
        'tb_logs/enhanced_dqn_run_1767145943.7603014_1',  # 0-390K (eval_freq=1000)
        'tb_logs/enhanced_dqn_run_1767236002.1943977_0'   # 390K-500K (eval_freq=10000)
    ]
    
    # 检查日志是否存在
    for log in enhanced_logs:
        if not os.path.exists(log):
            print(f"错误: 日志不存在 {log}")
            return
    
    print(f"合并训练日志:")
    print(f"  第一阶段 (0-390K, eval_freq=1000): {enhanced_logs[0]}")
    print(f"  第二阶段 (390K-500K, eval_freq=10000): {enhanced_logs[1]}")
    
    multi_hole_logs = glob.glob('tb_logs/multi_hole_dqn_run_*')
    
    # 创建输出目录
    os.makedirs('figures', exist_ok=True)
    
    # 1. 评估奖励曲线（合并）
    print("\n导出评估奖励曲线（合并0-500K）...")
    steps, rewards = merge_logs(enhanced_logs, 'eval/mean_reward')
    if steps:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, rewards, linewidth=2, color='#1f77b4')
        plt.xlabel('训练步数', fontsize=12)
        plt.ylabel('平均奖励', fontsize=12)
        plt.title('评估奖励变化曲线', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/eval_reward.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  保存到: figures/eval_reward.png")
    
    # 2. 评估episode长度（合并）
    print("导出评估episode长度曲线（合并0-500K）...")
    steps, lengths = merge_logs(enhanced_logs, 'eval/mean_ep_length')
    if steps:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, lengths, linewidth=2, color='#ff7f0e')
        plt.xlabel('训练步数', fontsize=12)
        plt.ylabel('平均Episode长度', fontsize=12)
        plt.title('评估Episode长度变化曲线', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/eval_length.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  保存到: figures/eval_length.png")
    
    # 3. 训练奖励曲线（合并）
    print("导出训练奖励曲线（合并0-500K）...")
    steps, train_rewards = merge_logs(enhanced_logs, 'rollout/ep_rew_mean')
    if steps:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_rewards, linewidth=2, color='#2ca02c', alpha=0.7)
        plt.xlabel('训练步数', fontsize=12)
        plt.ylabel('平均奖励', fontsize=12)
        plt.title('训练奖励变化曲线', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/train_reward.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  保存到: figures/train_reward.png")
    
    # 4. 探索率曲线（合并）
    print("导出探索率曲线（合并0-500K）...")
    steps, exploration = merge_logs(enhanced_logs, 'rollout/exploration_rate')
    if steps:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, exploration, linewidth=2, color='#d62728')
        plt.xlabel('训练步数', fontsize=12)
        plt.ylabel('探索率', fontsize=12)
        plt.title('探索率衰减曲线', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/exploration_rate.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  保存到: figures/exploration_rate.png")
    
    # 5. 多洞训练日志（合并多个stage）
    multi_hole_logs_list = [
        'tb_logs/multi_hole_dqn_run_1767324212.8868244_stage_1_0',
        'tb_logs/multi_hole_dqn_run_1767324212.8868244_stage_2_0',
        'tb_logs/multi_hole_dqn_run_1767493694.3129294_stage_1_0',
        'tb_logs/multi_hole_dqn_run_1767493694.3129294_stage_2_0'
    ]
    
    # 检查多洞日志是否存在
    existing_multi_logs = [log for log in multi_hole_logs_list if os.path.exists(log)]
    
    if existing_multi_logs:
        print(f"\n多洞训练日志（合并{len(existing_multi_logs)}个stage）:")
        for log in existing_multi_logs:
            print(f"  - {log}")
        
        # 多洞训练评估奖励（合并）
        print("\n导出多洞训练评估奖励曲线（合并）...")
        steps2, rewards2 = merge_logs(existing_multi_logs, 'eval/mean_reward')
        if steps2:
            plt.figure(figsize=(10, 6))
            plt.plot(steps2, rewards2, linewidth=2, color='#ff7f0e')
            plt.xlabel('训练步数', fontsize=12)
            plt.ylabel('平均奖励', fontsize=12)
            plt.title('多洞训练评估奖励变化曲线', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('figures/multi_hole_eval_reward.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  保存到: figures/multi_hole_eval_reward.png")
        
        # 对比图：单洞 vs 多洞
        print("导出单洞vs多洞对比图...")
        steps1, rewards1 = merge_logs(enhanced_logs, 'eval/mean_reward')
        
        if steps1 and steps2:
            plt.figure(figsize=(12, 6))
            plt.plot(steps1, rewards1, linewidth=2, label='单洞训练', color='#1f77b4')
            plt.plot(steps2, rewards2, linewidth=2, label='多洞训练', color='#ff7f0e')
            plt.xlabel('训练步数', fontsize=12)
            plt.ylabel('平均奖励', fontsize=12)
            plt.title('单洞训练 vs 多洞训练对比', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('figures/comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  保存到: figures/comparison.png")
    
    print("\n所有图表导出完成！")
    print("图片保存在 figures/ 目录下")


if __name__ == "__main__":
    plot_training_curves()
    print("\n所有图表导出完成！")
    print("图片保存在 figures/ 目录下")

if __name__ == "__main__":
    plot_training_curves()
