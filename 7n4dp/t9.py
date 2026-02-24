import sys

sys.path.append("..")

import os
import torch
import numpy as np
from src import *

if __name__ == '__main__':
    # 测试base_dim, embed_dim, 余弦退火
    set_seed(2026)
    create_dir()

    # === 修改点 1：定义 Train 和 Test 数据集路径 ===
    DATASET_TRAIN = "./data/train.json"
    DATASET_TEST = "./data/test.json"
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    TOTAL_EPISODES = 100000
    EVAL_INTERVAL = 10000

    generator = SOSDataGenerator(num_vars=7, degree=4)

    # === 修改点 2：初始化两个环境 ===
    # 训练环境：使用 train.json，限制采样 10000 条用于训练 (train_data_size=10000)
    env = SOSPruningEnvGT(generator, DATASET_TRAIN, device=DEVICE, train_data_size=10000)

    # 测试环境：使用 test.json，设一个很大的数以加载全部测试集 (train_data_size=100000)
    # 注意：env.py 的逻辑是 "如果数据量 > target_size 则采样"，设大一点就能全量加载
    test_env = SOSPruningEnvGT(generator, DATASET_TEST, device=DEVICE, train_data_size=10000)

    agent = DoubleDQNAgentPER(generator.coeff_dim, generator.mask_dim, device=DEVICE, base_dim=2048, embed_dim=64, total_episode=TOTAL_EPISODES)

    print("\n" + "=" * 60)
    print("STARTING RL TRAINING (With Train/Test Evaluation)")
    print(f"Train Set Size: {len(env.dataset)}")
    print(f"Test Set Size:  {len(test_env.dataset)}")
    print("=" * 60)

    # 记录训练数据
    rewards_history = []
    train_gap_history = []

    # === 修改点 3：分别记录 Train 和 Test 的评估结果 ===
    eval_train_gaps = []
    eval_test_gaps = []
    eval_epoch_history = []

    for episode in range(TOTAL_EPISODES+500000):
        # --- 1. 训练循环 (仅使用 env) ---
        state = env.reset()
        ep_reward = 0
        done = False

        while not done:
            coeffs, mask = state
            action = agent.select_action(coeffs, mask)

            # 训练步
            next_state, reward, done, _ = env.step(action, is_eval=False)

            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            ep_reward += reward

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        rewards_history.append(ep_reward)
        train_gap_history.append(env.min_feasible_size - env.current_gt_size)

        if episode % 20 == 0:
            agent.update_target_network()
        
        if agent.scheduler is not None and len(agent.memory) > agent.batch_size and episode < TOTAL_EPISODES:
            agent.scheduler.step()

        # --- 2. 日志打印 ---
        if (episode + 1) % 100 == 0:
            avg_r = np.mean(rewards_history[-100:])
            avg_train_gap = np.mean(train_gap_history[-100:])
            print(
                f"Ep {episode + 1:<5} | Train Avg R: {avg_r:6.2f} | Eps: {agent.epsilon:.3f} | Step Gap: {avg_train_gap:4.1f}")

        # --- 3. 定期评估 (分别评估 Train 和 Test) ---
        if (episode + 1) % EVAL_INTERVAL == 0:
            print(f"\n--- Evaluation at Episode {episode + 1} ---")

            # 评估 Train 集 (使用 env)
            print("[Eval Train Set]")
            gap_train = evaluate_agent(agent, env)

            # 评估 Test 集 (使用 test_env)
            print("[Eval Test Set]")
            gap_test = evaluate_agent(agent, test_env)

            eval_train_gaps.append(gap_train)
            eval_test_gaps.append(gap_test)
            eval_epoch_history.append(episode + 1)

            # === 修改点 4：调用更新后的绘图函数 ===
            plot_training_results(
                rewards_history,
                train_gap_history,
                eval_train_gaps,
                eval_test_gaps,
                eval_epoch_history,
                save_path=f'./picture/{os.path.splitext(os.path.basename(__file__))[0]}.png'
            )

            # 保存最佳模型 (以 Test Gap 为准，泛化能力更重要)
            if len(eval_test_gaps) > 1 and gap_test < min(eval_test_gaps[:-1]):
                print(f"[Save] New best TEST gap: {gap_test:.4f}. Saving model...")
                torch.save(agent.policy_net.state_dict(), f"./model/{os.path.splitext(os.path.basename(__file__))[0]}.pth")
            print("-" * 30 + "\n")

    print("Training Complete.")