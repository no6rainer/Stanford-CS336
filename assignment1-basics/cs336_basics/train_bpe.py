import regex as re
import os
from collections import Counter, defaultdict

from pretokenization_example import find_chunk_boundaries

pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "..", "data", "test.txt")
print(data_dir)

vocab_size = 1000

token_counts = Counter()
with open(data_dir, "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    print(boundaries)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        special_tokens = ["<|endoftext|>"]

        # Build a regex pattern that matches any of the special tokens
        special_pattern = "|".join(re.escape(tok) for tok in special_tokens)

        parts = re.split(special_pattern, chunk)

        for part in parts:
            for token in re.finditer(pretokenization_pattern, part):
                token_counts[token.group(0).encode("utf-8")] += 1

# Example element: ('e', 'st'): {('low', 'e', 'st'): 1, ('high', 'e', 'st'): 2, ......}
bytes_pair_map: dict[tuple[bytes, bytes], Counter[tuple[bytes, ...]]] = defaultdict(Counter)

# Example element: 'e': [('t', 'e'), ('e', 'st'), ......]
bytes_map: dict[bytes, list[tuple[bytes, bytes]]] = defaultdict(list)

for token, count in token_counts.items(): 
    token_bytes = tuple(token[i:i+1] for i in range(len(token)))
    for first, second in zip(token_bytes[:-1], token_bytes[1:]):
        bytes_pair = (first, second)
        bytes_pair_map[bytes_pair][token_bytes] += count
        bytes_map[first].append(bytes_pair)
        bytes_map[second].append(bytes_pair)

print(bytes_pair_map)