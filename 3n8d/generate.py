import sys

sys.path.append("..")

from src import *

if __name__ == '__main__':
    set_seed(2026)
    create_dir()
    num_vars = 3
    degree = 8
    gen = SOSDataGenerator(num_vars=num_vars, degree=degree)

    # Regenerate the dataset with minimal_mask
    data = gen.generate_dataset(total_samples=10000, ratio=(1, 1, 1, 0))
    gen.save_to_json(data, f"./data/train.json")

    data = gen.generate_dataset(total_samples=10000, ratio=(1, 1, 1, 0))
    gen.save_to_json(data, f"./data/test.json")