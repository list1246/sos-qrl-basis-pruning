# SOS 基底剪枝与 QRL 结果整理项目

本文档根据当前目录 `Thesis` 的实际文件重新整理。项目主要用于训练强化学习模型剪枝 SOS Gram 矩阵基底，并对真实多项式样例生成、验证和归档对应的 `Q_RL` 结果。

## 项目目标

对偶次多项式 `f(x)`，SOS 表示可写为：

```text
f(x) = z(x)^T Q z(x),  Q >= 0
```

其中 `z(x)` 是完整单项式基底，`Q` 是半正定 Gram 矩阵。本项目使用 Double DQN + Prioritized Experience Replay 训练剪枝策略，在保持 SDP 可行性的前提下尽量减少保留基底，得到：

```text
f(x) = z_RL(x)^T Q_RL z_RL(x)
```

当前项目还包含 20 个论文附录样例 `C1` 至 `C20` 的整理结果，每个样例目录中保存了对应的多项式、基准测试行和 `Q_RL` 文件。

## 当前目录结构

```text
Thesis/
├── PROJECT_README.md
├── train.xlsx
├── solve_sos_qrl.py
├── src/
│   ├── __init__.py
│   ├── generator.py
│   ├── env.py
│   ├── model.py
│   ├── agent.py
│   ├── buffer.py
│   └── util.py
├── all/
│   ├── bench_sdp.py
│   ├── basis_red_boxes.csv
│   ├── res.csv
│   ├── result_real.csv
│   ├── result_real.md
│   ├── qrl_results/
│   └── qrl_results_scs_check/
├── C1/ ... C20/
└── 2n6d/, 2n8d/, 2n10d/, 3n4d/, 3n6d/, 3n8d/,
    4n4d/, 4n6d/, 5n4d/, 6n4d/, 7n4d/, 8n2d/
```

## 核心代码

| 文件 | 作用 |
|---|---|
| `src/generator.py` | 构造 SOS 单项式基底、完整多项式系数空间和 `Q -> P` 系数映射，并生成训练/测试数据。 |
| `src/env.py` | `SOSPruningEnvGT` 强化学习环境，状态为系数向量和当前 mask，动作为删除一个基底项或 STOP。 |
| `src/model.py` | `TwoStreamQNet`，分别处理多项式系数流和 mask 流，输出每个剪枝动作及 STOP 的 Q 值。 |
| `src/agent.py` | `DoubleDQNAgentPER`，包含 Double DQN、epsilon-greedy、目标网络和 PER 经验回放。 |
| `src/buffer.py` | `SumTree` 与 `PrioritizedReplayBuffer`。 |
| `src/util.py` | 随机种子、目录创建、训练曲线绘制和 agent 评估工具。 |
| `solve_sos_qrl.py` | 读取多项式并求解缩减基底下的 `Q_RL`，输出 CSV、NPZ、JSON。 |
| `all/bench_sdp.py` | 对 `all/result_real.csv` 中样例运行 RL、Newton Polytope、TSSOS、随机剪枝和 SDP 计时。 |

## 实验目录

每个 `{n}n{d}d` 目录对应变量数 `n` 和多项式总次数 `d`，例如 `3n6d` 表示 3 个变量、6 次多项式。常见文件如下：

| 文件/目录 | 作用 |
|---|---|
| `generate.py` | 生成 `data/train.json` 和 `data/test.json`。 |
| `train.py` | 训练当前配置下的 RL 剪枝模型。 |
| `eval.py` | Newton Polytope 基线评估。 |
| `env_custom.py` | 部分目录中的自定义环境参数。 |
| `model/` | 训练得到的模型权重，常用文件名为 `train.pth`。 |
| `picture/` | 训练曲线图片。 |
| `result/` | 评估输出目录。 |

当前存在的实验配置：

```text
2n6d, 2n8d, 2n10d,
3n4d, 3n6d, 3n8d,
4n4d, 4n6d,
5n4d, 6n4d, 7n4d, 8n2d
```

## 运行环境

建议使用已有 Python/Conda 环境。项目依赖包括：

| 包 | 用途 |
|---|---|
| `torch` | 强化学习模型和训练。 |
| `numpy` | 数值计算。 |
| `sympy` | 多项式解析和展开。 |
| `cvxpy` | SDP 建模与求解。 |
| `scipy` | Newton Polytope 中的线性规划。 |
| `matplotlib` | 训练曲线绘图。 |
| `openpyxl` | 读取 Excel 数据集。 |

`bench_sdp.py` 默认使用 MOSEK 计时；`solve_sos_qrl.py` 默认使用 CLARABEL，并在需要时可切换到 SCS。

## 常用命令

在项目根目录运行：

```powershell
cd Thesis
```

生成某个配置的数据：

```powershell
cd 3n4d
python generate.py
```

训练某个配置的模型：

```powershell
cd 3n4d
python train.py
```

运行 Newton Polytope 基线：

```powershell
cd 3n4d
python eval.py
```

对真实样例求解 `Q_RL`：

```powershell
cd Thesis
python solve_sos_qrl.py --csv all\result_real.csv --all --outdir all\qrl_results
```

只求解单个样例：

```powershell
python solve_sos_qrl.py --csv all\result_real.csv --sample-id 33 --outdir all\qrl_results
```

用完整基底求解：

```powershell
python solve_sos_qrl.py --expr "x1**2 + x2**2" --nvars 2 --degree 2 --full-basis
```

运行综合基准测试：

```powershell
cd all
python bench_sdp.py
```

## `all/` 结果说明

| 文件/目录 | 内容 |
|---|---|
| `all/result_real.csv` | 真实多项式样例及完整基底、RL、NP、TSSOS 剪枝后基底数量。 |
| `all/result_real.md` | `result_real.csv` 的 Markdown 版本。 |
| `all/res.csv` | 综合基准测试结果，包含全基、RL、NP、TSSOS、随机剪枝的 SDP 时间和加速比。 |
| `all/qrl_results/` | 每个样例的 `Q_RL` 结果，包含 `*_Q_RL.csv`、`.npz`、`.json`。 |
| `all/qrl_results/summary.csv` | `Q_RL` 求解摘要：样本号、配置、基底大小、最小特征值、秩、残差和状态。 |
| `all/qrl_results_scs_check/` | 额外 SCS 检查结果；当前包含 sample 30。 |
| `all/basis_red_boxes.csv` | 基底缩减相关的辅助数据。 |

当前 `all/res.csv` 有 20 个样例。按现有结果统计：

| 指标 | 平均值 |
|---|---:|
| 完整基底数量 | 24.2 |
| RL 剪枝后 | 9.4 |
| NP 剪枝后 | 13.3 |
| TSSOS 剪枝后 | 13.7 |
| 随机剪枝后 | 21.2 |

当前 `all/qrl_results/summary.csv` 有 20 条 `optimal` 记录，最大重构残差约为 `2.68e-10`。

## `C1` 至 `C20` 归档目录

`C1` 到 `C20` 是按论文附录样例整理出的独立目录。每个目录包含：

| 文件/目录 | 内容 |
|---|---|
| `qrl_results/` | 该样例对应的 `Q_RL` CSV、NPZ、JSON。 |
| `res.csv` | `all/res.csv` 中该样例的单行基准测试结果。 |
| `result_real.csv` | `all/result_real.csv` 中该样例的单行多项式记录。 |
| `polynomial.txt` | 该样例的多项式文本和 sample id。 |
| `qrl_summary.csv` | `all/qrl_results/summary.csv` 中该样例的单行摘要。 |
| `manifest.json` | 样例编号、sample id、QRL 文件数量和额外检查标记。 |

样例映射如下：

| C 编号 | sample id |
|---|---:|
| C1 | 8 |
| C2 | 11 |
| C3 | 15 |
| C4 | 16 |
| C5 | 21 |
| C6 | 22 |
| C7 | 30 |
| C8 | 35 |
| C9 | 54 |
| C10 | 55 |
| C11 | 57 |
| C12 | 58 |
| C13 | 59 |
| C14 | 60 |
| C15 | 65 |
| C16 | 101 |
| C17 | 3 |
| C18 | 9 |
| C19 | 10 |
| C20 | 33 |

其中 `C7` 还包含 `qrl_results_scs_check/`，对应 sample 30 的额外 SCS 检查结果。

## `Q_RL` 输出格式

`solve_sos_qrl.py` 对每个样例输出三个文件：

| 文件 | 内容 |
|---|---|
| `sample_<id>_<config>_Q_RL.csv` | Gram 矩阵 `Q_RL` 的数值矩阵。 |
| `sample_<id>_<config>.npz` | 压缩数据包，包含 `Q_RL`、完整基底、RL 基底、active indices、秩、残差等。 |
| `sample_<id>_<config>.json` | 可读元数据，包含多项式、基底单项式、求解状态、最小特征值、秩和残差。 |

例如：

```text
all/qrl_results/sample_33_4n6d_Q_RL.csv
all/qrl_results/sample_33_4n6d.npz
all/qrl_results/sample_33_4n6d.json
```

## 模型配置

`solve_sos_qrl.py` 和 `all/bench_sdp.py` 使用同一套模型宽度配置：

```python
MODEL_CONFIG = {
    "2n6d":  {"base_dim": 512,  "embed_dim": 16},
    "2n8d":  {"base_dim": 512,  "embed_dim": 16},
    "2n10d": {"base_dim": 1024, "embed_dim": 32},
    "3n4d":  {"base_dim": 128,  "embed_dim": 8},
    "3n6d":  {"base_dim": 256,  "embed_dim": 8},
    "3n8d":  {"base_dim": 1024, "embed_dim": 32},
    "4n4d":  {"base_dim": 256,  "embed_dim": 8},
    "4n6d":  {"base_dim": 1024, "embed_dim": 32},
    "5n4d":  {"base_dim": 256,  "embed_dim": 8},
    "6n4d":  {"base_dim": 1024, "embed_dim": 32},
    "7n4d":  {"base_dim": 4096, "embed_dim": 256},
    "8n2d":  {"base_dim": 128,  "embed_dim": 8},
}
```

模型权重默认从对应目录读取：

```text
<config>/model/train.pth
```

例如 `3n6d/model/train.pth`。

## 注意事项

- 各实验目录脚本通常通过 `sys.path.append("..")` 引用 `src/`，建议在对应实验目录内运行 `generate.py`、`train.py`、`eval.py`。
- `bench_sdp.py` 使用 `all/result_real.csv`，并假设对应配置目录下存在训练好的 `model/train.pth`。
- 如果没有 MOSEK，`bench_sdp.py` 中的 MOSEK 计时会失败；可改用 `solve_sos_qrl.py --solver CLARABEL` 或 `--solver SCS` 做单样例验证。
- 项目中部分源码注释存在历史编码问题，但核心 Python 标识符和当前 README 均按 UTF-8 使用。
