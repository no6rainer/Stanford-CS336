import regex as re
import os
from collections import Counter, defaultdict
import heapq

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

bytes_pair_count: Counter[tuple[bytes, bytes]] = Counter()

# Example element: ('e', 'st'): {('low', 'e', 'st'): 1, ('high', 'e', 'st'): 2, ......}
bytes_pair_map: dict[tuple[bytes, bytes], Counter[tuple[bytes, ...]]] = defaultdict(Counter)

# TODO: classify each bytes list into two lists where the key is the first and the second element respectively
# Example element: 'e': [('t', 'e'), ('e', 'st'), ......]
bytes_map: dict[bytes, list[tuple[bytes, bytes]]] = defaultdict(list)

for token, count in token_counts.items(): 
    token_bytes = tuple(token[i:i+1] for i in range(len(token)))
    for first, second in zip(token_bytes[:-1], token_bytes[1:]):
        bytes_pair = (first, second)
        bytes_pair_count[bytes_pair] += count
        bytes_pair_map[bytes_pair][token_bytes] += count
        bytes_map[first].append(bytes_pair)
        bytes_map[second].append(bytes_pair)

bp_heap: list[tuple[int, tuple[bytes, bytes]]] = []

def push_pair(pair: tuple[bytes, bytes], weight: int) -> None:
    heapq.heappush(bp_heap, (-weight, pair))

def pop_max() -> tuple[tuple[bytes, bytes], int]:
    neg_weight, pair = heapq.heappop(bp_heap)
    return pair, -neg_weight

for pair, count in bytes_pair_count.items():
    push_pair(pair, count)

vocab: dict[int, bytes] = defaultdict(bytes)
merges: list[tuple[bytes, bytes]] = []

curr_vocab_size = 256
while curr_vocab_size < vocab_size:
    # TODO: Handle the case with multiple maxes and take the lexicographically greater pair
    pair, count = pop_max()

    if count != bytes_pair_count[pair]:
        continue

    first, second = pair
    vocab[curr_vocab_size] = first + second
    merges.append(pair)

    first_pairs = []
    for neighbor_pair in bytes_map[first]:
        _, neighbor_second = neighbor_pair
        if first == neighbor_second:
            first_pairs.append(neighbor_pair)

    second_pairs = []
    for neighbor_pair in bytes_map[second]:
        neighbor_first, _ = neighbor_pair
        if second == neighbor_first:
            second_pairs.append(neighbor_pair)

print(bytes_pair_map)