from cs336_basics import train_bpe
import os

if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data", "assignment_example.txt")
    print(train_bpe(data_dir, 10000, ["<|endoftext|>"]))