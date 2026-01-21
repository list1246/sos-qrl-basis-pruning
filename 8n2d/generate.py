import sys

sys.path.append("..")

from src import *

if __name__ == '__main__':
    set_seed(2026)
    create_dir()
    num_vars = 8
    degree = 2
    gen = SOSDataGenerator(num_vars=num_vars, degree=degree)

    # 重新生成带 minimal_mask 的数据集
    data = gen.generate_dataset(total_samples=60000)
    gen.save_to_json(data, f"./data/train.json")

    data = gen.generate_dataset(total_samples=10000, ratio=(1, 1, 1, 0))
    gen.save_to_json(data, f"./data/test.json")