import regex as re
import os
import mmap
from collections import Counter, defaultdict
import heapq
from multiprocessing import Pool
from functools import partial

from .pretokenization_example import find_chunk_boundaries

_shared_mmap: mmap.mmap | None = None
_pretok_re: re.Pattern | None = None
_split_re: re.Pattern | None = None

def _init_worker(path: str, special_tokens: list[str]):
    global _shared_mmap, _pretok_re, _split_re

    fd = os.open(path, os.O_RDONLY)
    _shared_mmap = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    os.close(fd)

    pretok_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    _pretok_re = re.compile(pretok_pattern)

    split_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    _split_re = re.compile(split_pattern)

def tokenize_chunk(start: int, end: int) -> Counter[bytes]:
    global _shared_mmap, _pretok_re, _split_re

    assert _shared_mmap is not None, "worker not initialized"
    assert _pretok_re is not None, "worker not initialized"
    assert _split_re is not None, "worker not initialized"

    token_counts = Counter()

    mv = memoryview(_shared_mmap)[start:end]
    chunk = mv.tobytes().decode("utf-8", errors="ignore")

    parts = _split_re.split(chunk)

    for part in parts:
        for token in _pretok_re.finditer(part):
            token_counts[token.group(0).encode("utf-8")] += 1

    return token_counts

def _reduce_counters(counters):
    out = Counter()
    for c in counters:
        out.update(c)
    return out

def train_bpe(input_path: str | os.PathLike, 
              vocab_size: int, 
              special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    with open(input_path, "rb") as f:
        num_processes = 16
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

    with Pool(
        processes=16,
        initializer=_init_worker,
        initargs=(input_path, special_tokens),
    ) as pool:
        counts = pool.starmap(tokenize_chunk, tasks)

    token_counts = _reduce_counters(counts)

    # Example element: ('e', 'st'): 2
    pair_count: Counter[tuple[bytes, bytes]] = Counter()

    # Example element: 'lowest': ('low', 'e', 'st')
    token_bytes_map: dict[bytes, tuple[bytes, ...]] = defaultdict(tuple)

    # Example element: ('e', 'st'): {'lowest', 'highest', ...}
    pair_token_map: dict[tuple[bytes, bytes], set[bytes]] = defaultdict(set)

    for token, count in token_counts.items(): 
        token_bytes = tuple(token[i:i + 1] for i in range(len(token)))
        token_bytes_map[token] = token_bytes
        for first, second in zip(token_bytes[:-1], token_bytes[1:]):
            pair = (first, second)
            pair_token_map[pair].add(token)
            pair_count[pair] += count

    bp_heap: list[tuple[int, tuple[bytes, bytes]]] = []

    def push_pair(pair: tuple[bytes, bytes], weight: int) -> None:
        heapq.heappush(bp_heap, (-weight, pair))

    def pop_max() -> tuple[tuple[bytes, bytes], int]:
        neg_weight, pair = heapq.heappop(bp_heap)
        return pair, -neg_weight

    for pair, count in pair_count.items():
        push_pair(pair, count)

    vocab: dict[int, bytes] = defaultdict(bytes)
    merges: list[tuple[bytes, bytes]] = []
    
    for i in range(256):
        vocab[i] = bytes([i])

    curr_vocab_size = 256
    while bp_heap and curr_vocab_size < vocab_size:
        # handle tied max pairs and choose the lexicographically greatest pair
        next_pair, max_count = pop_max()
        tied = [next_pair]

        while bp_heap:
            next_pair, next_count = pop_max()
            if next_count != max_count:
                push_pair(next_pair, next_count)
                break
            else:
                tied.append(next_pair)

        pair = max(tied)
        tied.remove(pair)
        for tied_pair in tied:
            push_pair(tied_pair, max_count)

        count = max_count
        
        # filter stale pairs
        if count != pair_count[pair]:
            continue

        first, second = pair
        new_byte = first + second
        vocab[curr_vocab_size] = new_byte
        merges.append(pair)
        # print(f"merging: {pair} -> {new_byte}")

        token_set = pair_token_map[pair]

        changed_pairs = set()

        for token in token_set:
            token_count = token_counts[token]
            token_bytes = token_bytes_map[token]
            new_token_bytes = []

            token_pair_count = Counter()

            # initial scan to record pair count
            for temp_pair in zip(token_bytes[:-1], token_bytes[1:]):
                token_pair_count[temp_pair] += 1

            i = 0
            while i < len(token_bytes) - 1:
                # scan the token for matches
                if token_bytes[i] == first and token_bytes[i + 1] == second:

                    # handle changes of the left pair
                    if i - 1 >= 0:
                        left_pair = (token_bytes[i - 1], first)
                        pair_count[left_pair] -= token_count
                        token_pair_count[left_pair] -= 1
                        changed_pairs.add(left_pair)

                        new_left_pair = (token_bytes[i - 1], new_byte)
                        pair_count[new_left_pair] += token_count
                        pair_token_map[new_left_pair].add(token)
                        changed_pairs.add(new_left_pair)

                    # handle changes of the right pair
                    if i + 2 <= len(token_bytes) - 1:
                        right_pair = (second, token_bytes[i + 2])
                        pair_count[right_pair] += token_count
                        token_pair_count[right_pair] -= 1
                        changed_pairs.add(right_pair)

                        new_right_pair = (new_byte, token_bytes[i + 2])
                        pair_count[new_right_pair] += token_count
                        pair_token_map[new_right_pair].add(token)
                        changed_pairs.add(new_right_pair)

                    new_token_bytes.append(new_byte)

                    # if first != second: impossible for i + 1 pair to be a match 
                    # if first == second: skip the i + 1 pair to avoid duplicate counting
                    # and ensure that new_token_bytes is formed correctly by skipping the merged pair
                    i += 2

                else:
                    new_token_bytes.append(token_bytes[i])
                    if i == len(token_bytes) - 2:
                        new_token_bytes.append(token_bytes[i + 1])

                    i += 1

            # register the new token bytes
            new_token_bytes_tuple = tuple(new_token_bytes)
            token_bytes_map[token] = new_token_bytes_tuple

            # delete the pairs that no longer exists in token
            for temp_pair, temp_count in token_pair_count.items():
                if temp_count == 0:
                    pair_token_map[temp_pair].discard(token)

        # update the heap
        for changed_pair in changed_pairs:
            push_pair(changed_pair, pair_count[changed_pair])

        curr_vocab_size += 1
    
    return vocab, merges