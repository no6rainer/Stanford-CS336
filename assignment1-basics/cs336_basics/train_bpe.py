import regex as re
import os
import mmap
from collections import Counter, defaultdict
import heapq
from multiprocessing import Pool
from dataclasses import dataclass

from .pretokenization_example import find_chunk_boundaries

_shared_mmap: mmap.mmap | None = None
_pretok_re: re.Pattern | None = None
_split_re: re.Pattern | None = None


@dataclass(frozen=True)
class HeapItem:
    count: int
    pair: tuple[bytes, bytes]

    def __lt__(self, other: "HeapItem") -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


def _init_worker(path: str, special_tokens: list[str]):
    global _shared_mmap, _pretok_re, _split_re

    fd = os.open(path, os.O_RDONLY)
    _shared_mmap = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    os.close(fd)

    pretok_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    _pretok_re = re.compile(pretok_pattern)

    if special_tokens:
        # Sort by length to avoid partial matches when special tokens overlap.
        split_pattern = "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
        _split_re = re.compile(split_pattern)
    else:
        _split_re = None

def tokenize_chunk(start: int, end: int) -> Counter[bytes]:
    global _shared_mmap, _pretok_re, _split_re

    assert _shared_mmap is not None, "worker not initialized"
    assert _pretok_re is not None, "worker not initialized"

    token_counts = Counter()

    mv = memoryview(_shared_mmap)[start:end]
    chunk = mv.tobytes().decode("utf-8", errors="ignore")

    parts = _split_re.split(chunk) if _split_re is not None else [chunk]

    for part in parts:
        for token in _pretok_re.finditer(part):
            token_counts[token.group(0).encode("utf-8")] += 1

    return token_counts

def _reduce_counters(counters):
    out = Counter()
    for c in counters:
        out.update(c)
    return out

def train_bpe(
    input_path: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    num_processes = os.cpu_count() or 1

    with open(input_path, "rb") as f:
        if special_tokens:
            # Keep chunk boundaries on a special token boundary to avoid merges across it.
            split_token = special_tokens[0].encode("utf-8")
            boundaries = find_chunk_boundaries(f, num_processes, split_token)
        else:
            f.seek(0, os.SEEK_END)
            boundaries = [0, f.tell()]

    tasks = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

    with Pool(
        processes=num_processes,
        initializer=_init_worker,
        initargs=(input_path, special_tokens),
    ) as pool:
        counts = pool.starmap(tokenize_chunk, tasks)

    token_counts = _reduce_counters(counts)

    # Example element: ('e', 'st'): 2
    pair_count: Counter[tuple[bytes, bytes]] = Counter()

    # Example element: 'lowest': ('low', 'e', 'st')
    token_bytes_map: dict[bytes, tuple[bytes, ...]] = {}

    # Example element: ('e', 'st'): {'lowest', 'highest', ...}
    pair_token_map: dict[tuple[bytes, bytes], set[bytes]] = defaultdict(set)

    for token, count in token_counts.items(): 
        token_bytes = tuple(token[i:i + 1] for i in range(len(token)))
        token_bytes_map[token] = token_bytes
        for first, second in zip(token_bytes[:-1], token_bytes[1:]):
            pair = (first, second)
            pair_token_map[pair].add(token)
            pair_count[pair] += count

    bp_heap: list[HeapItem] = []

    def push_pair(pair: tuple[bytes, bytes], weight: int) -> None:
        heapq.heappush(bp_heap, HeapItem(weight, pair))

    def pop_max() -> tuple[tuple[bytes, bytes], int]:
        item = heapq.heappop(bp_heap)
        return item.pair, item.count

    for pair, count in pair_count.items():
        push_pair(pair, count)

    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    
    for i, tok in enumerate(special_tokens):
        vocab[i] = tok.encode("utf-8")

    byte_offset = len(special_tokens)
    for b in range(256):
        vocab[byte_offset + b] = bytes([b])

    curr_vocab_size = byte_offset + 256

    while bp_heap and curr_vocab_size < vocab_size:
        pair, count = pop_max()

        # No mergeable pair remains
        if count <= 0:
            break
        
        # filter stale pairs
        if count != pair_count[pair]:
            continue

        first, second = pair
        new_byte = first + second
        vocab[curr_vocab_size] = new_byte
        merges.append(pair)

        token_set = list(pair_token_map[pair])

        changed_pairs = set()

        for token in token_set:
            token_count = token_counts[token]
            token_bytes = token_bytes_map[token]
            new_token_bytes = []
            removed_pairs = set()

            i = 0
            while i < len(token_bytes):
                # scan the token for matches
                if i + 1 < len(token_bytes) and token_bytes[i] == first and token_bytes[i + 1] == second:
                    pair_count[pair] -= token_count
                    changed_pairs.add(pair)
                    removed_pairs.add(pair)

                    # handle changes of the left pair
                    if new_token_bytes:
                        left_pair = (new_token_bytes[-1], first)
                        pair_count[left_pair] -= token_count
                        changed_pairs.add(left_pair)
                        removed_pairs.add(left_pair)

                        new_left_pair = (new_token_bytes[-1], new_byte)
                        pair_count[new_left_pair] += token_count
                        pair_token_map[new_left_pair].add(token)
                        changed_pairs.add(new_left_pair)

                    # handle changes of the right pair
                    if i + 2 < len(token_bytes):
                        right_pair = (second, token_bytes[i + 2])
                        pair_count[right_pair] -= token_count
                        changed_pairs.add(right_pair)
                        removed_pairs.add(right_pair)

                        new_right_pair = (new_byte, token_bytes[i + 2])
                        pair_count[new_right_pair] += token_count
                        pair_token_map[new_right_pair].add(token)
                        changed_pairs.add(new_right_pair)

                    new_token_bytes.append(new_byte)
                    i += 2

                else:
                    new_token_bytes.append(token_bytes[i])
                    i += 1

            # register the new token bytes
            new_token_bytes_tuple = tuple(new_token_bytes)
            token_bytes_map[token] = new_token_bytes_tuple

            # Remove membership only for pairs touched by this merge pass.
            new_pair_set = set(zip(new_token_bytes_tuple[:-1], new_token_bytes_tuple[1:]))
            for removed_pair in removed_pairs:
                if removed_pair not in new_pair_set:
                    pair_token_map[removed_pair].discard(token)

        # update the heap
        for changed_pair in changed_pairs:
            push_pair(changed_pair, pair_count[changed_pair])

        curr_vocab_size += 1
    
    return vocab, merges
