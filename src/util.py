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
    Fix all random seeds to make experiments reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs

    # Ensure CuDNN is deterministic as well
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Info] Random seed set to {seed}")


# def plot_training_results(rewards, training_gaps, eval_gaps, eval_epochs, window_size=200,
#                           save_path="training_plot.png"):
#     # === Change 1:Adjust canvas size ===
#     # Width 16 (ensure the x-axis is long enough), height 18 (ensure three vertically stacked plots are not flattened)
#     plt.figure(figsize=(16, 18))
#
#     # Set the global font size to prevent fonts from being too small in large figures
#     plt.rcParams.update({'font.size': 12})
#
#     # ==========================================
#     # Subplot 1: Training Gap (smoothed)
#     # ==========================================
#     # === Change 2:Change layout to 3 rows by 1 column, first plot ===
#     plt.subplot(3, 1, 1)
#     if len(training_gaps) > 0:
#         # A translucent background of raw data can be kept depending on requirements
#         # plt.plot(training_gaps, alpha=0.15, color='gray', label='Raw Gap')
#
#         if len(training_gaps) >= window_size:
#             weights = np.ones(window_size) / window_size
#             gap_ma = np.convolve(training_gaps, weights, mode='valid')
#             # Adjust the x-axis to align it
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
#     # Subplot 2: Reward (smoothed)
#     # ==========================================
#     # === Change 2:Change layout to 3 rows by 1 column, second plot ===
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
#     # Subplot 3: Evaluation Gap (average Gap on the test set)
#     # ==========================================
#     # === Change 2:Change layout to 3 rows by 1 column, third plot ===
#     plt.subplot(3, 1, 3)
#     if len(eval_gaps) > 0:
#         plt.plot(eval_epochs, eval_gaps, marker='o', color='green', linewidth=2, markersize=6, label='Eval Avg Gap')
#
#         # === Change 3:Optimize annotation display to avoid crowding ===
#         # Logic: if there are too many points, label every Nth point, but always label the minimum and final values
#         total_points = len(eval_gaps)
#
#         # Dynamically compute the interval so the plot has at most about 20 annotations and avoids overlap
#         step = max(1, total_points // 20)
#
#         min_val = min(eval_gaps)
#         min_idx = eval_gaps.index(min_val)
#
#         for i in range(total_points):
#             # Only annotate the starting point, every step-th point, the minimum point, and the last point
#             if i == 0 or i == total_points - 1 or i == min_idx or i % step == 0:
#                 txt = f"{eval_gaps[i]:.2f}"
#
#                 # Special-case the minimum point and show it in bold
#                 weight = 'bold' if i == min_idx else 'normal'
#                 color = 'red' if i == min_idx else 'black'
#
#                 # Slightly offset the position
#                 xy_text = (0, -15) if i % (step * 2) == 0 else (0, 10)
#                 if i == min_idx: xy_text = (0, -20)  # Place the minimum point slightly lower
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
#     # Automatically adjust the layout to prevent titles and axes from overlapping
#     plt.tight_layout(pad=3.0)
#
#     plt.savefig(save_path)
#     print(f"[Plot] Training curves saved to {save_path}")
#     plt.close()

def plot_training_results(rewards, training_gaps, eval_train_gaps, eval_test_gaps, eval_epochs, window_size=200,
                          save_path="training_plot.png"):
    plt.figure(figsize=(16, 18))
    plt.rcParams.update({'font.size': 12})

    # Subplot 1: Training Gap
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

    # Subplot 2: Reward
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

    # === Change:Subplot 3 plots both Train and Test evaluation curves ===
    plt.subplot(3, 1, 3)

    # Helper function: plot curves with sparse annotations
    def plot_with_annotations(epochs, gaps, color, label, offset_base):
        if len(gaps) == 0: return
        plt.plot(epochs, gaps, marker='o', color=color, linewidth=2, markersize=6, label=label)

        total_points = len(gaps)
        step = max(1, total_points // 15)  # sparse annotations
        min_val = min(gaps)
        min_idx = gaps.index(min_val)

        for i in range(total_points):
            if i == 0 or i == total_points - 1 or i == min_idx or i % step == 0:
                txt = f"{gaps[i]:.2f}"
                is_min = (i == min_idx)
                weight = 'bold' if is_min else 'normal'
                # Offset Train and Test annotation positions to prevent overlap
                y_offset = offset_base if not is_min else (offset_base - 10 if offset_base < 0 else offset_base + 10)
                xy_text = (0, y_offset)

                plt.annotate(txt, (epochs[i], gaps[i]), textcoords="offset points", xytext=xy_text,
                             ha='center', fontsize=9, color=color, fontweight=weight)

    # Plot Train Eval (green, annotations below)
    plot_with_annotations(eval_epochs, eval_train_gaps, 'green', 'Eval Train Gap', -15)
    # Plot Test Eval (orange, annotations above)
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
    Iterate through the entire Dataset in the environment and test with a greedy policy.
    Rules:
    1. Epsilon = 0 (no randomness)
    2. Step is_eval=True (stop immediately when a critical term is pruned)
    Return: average Gap
    """
    print(">>> Starting Evaluation on Full Dataset...")

    # Save the original epsilon and restore it after testing
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    total_gap = 0
    dataset_len = len(env.dataset)

    # Iterate through each data record
    for i in range(dataset_len):
        state = env.reset(sample_idx=i)
        done = False

        while not done:
            coeffs, mask = state
            action = agent.select_action(coeffs, mask)

            # Use step in eval mode
            next_state, _, done, _ = env.step(action, is_eval=True)
            state = next_state

        # Record the Gap for this episode (min_feasible_size - gt_size)
        # Note: in is_eval=True mode, pruning a critical term directly ends the episode, and mask remains in the pre-error feasible state
        # Therefore min_feasible_size is the limit the Agent can reach
        gap = env.min_feasible_size - env.current_gt_size
        total_gap += gap

    avg_gap = total_gap / dataset_len
    print(f">>> Evaluation Complete. Avg Gap: {avg_gap:.4f}")

    # Restore epsilon
    agent.epsilon = original_epsilon
    return avg_gap
