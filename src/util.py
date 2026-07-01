import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import os


def create_dir():
    os.makedirs('./model', exist_ok=True)
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./picture', exist_ok=True)
    os.makedirs('./result', exist_ok=True)


def set_seed(seed=2026):
    """
    固定所有随机种子，保证实验可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

    # 保证 CuDNN 也是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Info] Random seed set to {seed}")


# def plot_training_results(rewards, training_gaps, eval_gaps, eval_epochs, window_size=200,
#                           save_path="training_plot.png"):
#     # === 修改点 1：调整画布大小 ===
#     # 宽 16 (保证横坐标够长), 高 18 (保证竖向排列 3 张图不扁)
#     plt.figure(figsize=(16, 18))
#
#     # 设置全局字体大小，防止大图字体太小
#     plt.rcParams.update({'font.size': 12})
#
#     # ==========================================
#     # 子图 1: Training Gap (平滑)
#     # ==========================================
#     # === 修改点 2：布局改为 3行1列，第1张 ===
#     plt.subplot(3, 1, 1)
#     if len(training_gaps) > 0:
#         # 可以保留原始数据的半透明背景，看具体需求
#         # plt.plot(training_gaps, alpha=0.15, color='gray', label='Raw Gap')
#
#         if len(training_gaps) >= window_size:
#             weights = np.ones(window_size) / window_size
#             gap_ma = np.convolve(training_gaps, weights, mode='valid')
#             # 调整 x 轴，使其对齐
#             ma_x = range(window_size - 1, len(training_gaps))
#             plt.plot(ma_x, gap_ma, color='red', linewidth=2, label=f'Train Gap (MA {window_size})')
#
#     plt.title("Training Gap Process", fontsize=14, fontweight='bold')
#     plt.xlabel("Episode")
#     plt.ylabel("Gap")
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend(loc='upper right')
#
#     # ==========================================
#     # 子图 2: Reward (平滑)
#     # ==========================================
#     # === 修改点 2：布局改为 3行1列，第2张 ===
#     plt.subplot(3, 1, 2)
#     if len(rewards) > 0:
#         # plt.plot(rewards, alpha=0.15, color='lightblue', label='Raw Reward')
#
#         if len(rewards) >= window_size:
#             weights = np.ones(window_size) / window_size
#             reward_ma = np.convolve(rewards, weights, mode='valid')
#             ma_x = range(window_size - 1, len(rewards))
#             plt.plot(ma_x, reward_ma, color='blue', linewidth=2, label=f'Train Reward (MA {window_size})')
#
#     plt.title("Episode Reward Process", fontsize=14, fontweight='bold')
#     plt.xlabel("Episode")
#     plt.ylabel("Reward")
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend(loc='lower right')
#
#     # ==========================================
#     # 子图 3: Evaluation Gap (测试集平均 Gap)
#     # ==========================================
#     # === 修改点 2：布局改为 3行1列，第3张 ===
#     plt.subplot(3, 1, 3)
#     if len(eval_gaps) > 0:
#         plt.plot(eval_epochs, eval_gaps, marker='o', color='green', linewidth=2, markersize=6, label='Eval Avg Gap')
#
#         # === 修改点 3：优化标注显示，防止拥挤 ===
#         # 逻辑：如果点太多，就每隔 N 个点标一个数字，但必须标出最小值和最后一个值
#         total_points = len(eval_gaps)
#
#         # 动态计算间隔：保证图上最多只有约 20 个标注，防止重叠
#         step = max(1, total_points // 20)
#
#         min_val = min(eval_gaps)
#         min_idx = eval_gaps.index(min_val)
#
#         for i in range(total_points):
#             # 只标注：起步点、每隔 step 的点、最低点、最后一点
#             if i == 0 or i == total_points - 1 or i == min_idx or i % step == 0:
#                 txt = f"{eval_gaps[i]:.2f}"
#
#                 # 特殊处理最低点，用粗体显示
#                 weight = 'bold' if i == min_idx else 'normal'
#                 color = 'red' if i == min_idx else 'black'
#
#                 # 稍微错开一下位置
#                 xy_text = (0, -15) if i % (step * 2) == 0 else (0, 10)
#                 if i == min_idx: xy_text = (0, -20)  # 最低点往下放一点
#
#                 plt.annotate(txt,
#                              (eval_epochs[i], eval_gaps[i]),
#                              textcoords="offset points",
#                              xytext=xy_text,
#                              ha='center',
#                              fontsize=10,
#                              color=color,
#                              fontweight=weight,
#                              arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5) if i == min_idx else None)
#
#     plt.title("Evaluation Gap (Greedy Strategy on Full Dataset)", fontsize=14, fontweight='bold')
#     plt.xlabel("Episode")
#     plt.ylabel("Avg Gap")
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend()
#
#     # 自动调整布局，防止标题和坐标轴重叠
#     plt.tight_layout(pad=3.0)
#
#     plt.savefig(save_path)
#     print(f"[Plot] Training curves saved to {save_path}")
#     plt.close()

def plot_training_results(rewards, training_gaps, eval_train_gaps, eval_test_gaps, eval_epochs, window_size=200,
                          save_path="training_plot.png"):
    plt.figure(figsize=(16, 18))
    plt.rcParams.update({'font.size': 12})

    # 子图 1: Training Gap
    plt.subplot(3, 1, 1)
    if len(training_gaps) > 0:
        if len(training_gaps) >= window_size:
            weights = np.ones(window_size) / window_size
            gap_ma = np.convolve(training_gaps, weights, mode='valid')
            ma_x = range(window_size - 1, len(training_gaps))
            plt.plot(ma_x, gap_ma, color='red', linewidth=2, label=f'Train Step Gap (MA {window_size})')
    plt.title("Training Gap Process", fontsize=14, fontweight='bold')
    plt.xlabel("Episode")
    plt.ylabel("Gap")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')

    # 子图 2: Reward
    plt.subplot(3, 1, 2)
    if len(rewards) > 0:
        if len(rewards) >= window_size:
            weights = np.ones(window_size) / window_size
            reward_ma = np.convolve(rewards, weights, mode='valid')
            ma_x = range(window_size - 1, len(rewards))
            plt.plot(ma_x, reward_ma, color='blue', linewidth=2, label=f'Train Reward (MA {window_size})')
    plt.title("Episode Reward Process", fontsize=14, fontweight='bold')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')

    # === 修改点：子图 3 同时绘制 Train 和 Test 的评估曲线 ===
    plt.subplot(3, 1, 3)

    # 辅助函数：绘制带稀疏标注的曲线
    def plot_with_annotations(epochs, gaps, color, label, offset_base):
        if len(gaps) == 0: return
        plt.plot(epochs, gaps, marker='o', color=color, linewidth=2, markersize=6, label=label)

        total_points = len(gaps)
        step = max(1, total_points // 15)  # 稀疏标注
        min_val = min(gaps)
        min_idx = gaps.index(min_val)

        for i in range(total_points):
            if i == 0 or i == total_points - 1 or i == min_idx or i % step == 0:
                txt = f"{gaps[i]:.2f}"
                is_min = (i == min_idx)
                weight = 'bold' if is_min else 'normal'
                # 错开 Train 和 Test 的标注位置，防止重叠
                y_offset = offset_base if not is_min else (offset_base - 10 if offset_base < 0 else offset_base + 10)
                xy_text = (0, y_offset)

                plt.annotate(txt, (epochs[i], gaps[i]), textcoords="offset points", xytext=xy_text,
                             ha='center', fontsize=9, color=color, fontweight=weight)

    # 绘制 Train Eval (绿色，标注在下方)
    plot_with_annotations(eval_epochs, eval_train_gaps, 'green', 'Eval Train Gap', -15)
    # 绘制 Test Eval (橙色，标注在上方)
    plot_with_annotations(eval_epochs, eval_test_gaps, 'orange', 'Eval Test Gap', 10)

    plt.title("Evaluation Gap (Train Subset vs Test Full)", fontsize=14, fontweight='bold')
    plt.xlabel("Episode")
    plt.ylabel("Avg Gap")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path)
    print(f"[Plot] Training curves saved to {save_path}")
    plt.close()


def evaluate_agent(agent, env):
    """
    遍历环境中的整个 Dataset，使用 greedy 策略进行测试。
    规则：
    1. Epsilon = 0 (无随机)
    2. Step is_eval=True (剪错关键项直接停止)
    返回：平均 Gap
    """
    print(">>> Starting Evaluation on Full Dataset...")

    # 保存原始 epsilon，测试完恢复
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    total_gap = 0
    dataset_len = len(env.dataset)

    # 遍历每一条数据
    for i in range(dataset_len):
        state = env.reset(sample_idx=i)
        done = False

        while not done:
            coeffs, mask = state
            action = agent.select_action(coeffs, mask)

            # 使用 eval 模式的 step
            next_state, _, done, _ = env.step(action, is_eval=True)
            state = next_state

        # 记录本局的 Gap (min_feasible_size - gt_size)
        # 注意：在 is_eval=True 模式下，如果剪错会直接 Done，mask 保持在剪错前的状态（可行状态）
        # 所以 min_feasible_size 就是 Agent 能够达到的极限
        gap = env.min_feasible_size - env.current_gt_size
        total_gap += gap

    avg_gap = total_gap / dataset_len
    print(f">>> Evaluation Complete. Avg Gap: {avg_gap:.4f}")

    # 恢复 epsilon
    agent.epsilon = original_epsilon
    return avg_gap
