from pathlib import Path
from time import perf_counter
import os

from cs336_basics.tokenizer import Tokenizer

input_path = Path("data/owt_train.txt")
num_bytes = input_path.stat().st_size

tokenizer = Tokenizer.from_files(
    "artifacts/owt_vocab_32000.json",
    "artifacts/owt_merges_32000.txt",
    ["<|endoftext|>"],
)

start = perf_counter()
token_ids = tokenizer.efficient_encode_file_multiprocess(
    input_path,
    num_processes=os.cpu_count() or 1,
)
elapsed = perf_counter() - start
num_tokens = int(token_ids.size)

throughput_bps = num_bytes / elapsed
pile_seconds = 825e9 / throughput_bps
pile_hours = pile_seconds / 3600

print(f"elapsed: {elapsed:.2f} s")
print(f"throughput: {throughput_bps / 1e6:.2f} MB/s")
print(f"tokens: {num_tokens}")
print(f"estimated time for 825 GB: {pile_hours:.2f} hours")
