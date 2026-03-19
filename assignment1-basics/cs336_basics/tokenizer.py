from __future__ import annotations

import heapq
import json
from collections.abc import Iterable, Iterator
import mmap
import os
from multiprocessing import Pool

import numpy as np
import regex as re

from .pretokenization_example import find_chunk_boundaries


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Tokenizer:
    _efficient_shared_mmap: mmap.mmap | None = None
    _efficient_worker_tokenizer: Tokenizer | None = None

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []

        vocab_values = set(self.vocab.values())
        next_token_id = len(self.vocab)
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in vocab_values:
                self.vocab[next_token_id] = special_token_bytes
                vocab_values.add(special_token_bytes)
                next_token_id += 1

        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self.special_token_to_id = {
            special_token: self.bytes_to_id[special_token.encode("utf-8")]
            for special_token in self.special_tokens
        }
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        pretok_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._pretok_re = re.compile(pretok_pattern)

        if self.special_tokens:
            split_pattern = "|".join(
                re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)
            )
            self._special_token_re = re.compile(f"({split_pattern})")
            self._special_token_set = set(self.special_tokens)
        else:
            self._special_token_re = None
            self._special_token_set = set()

        self._efficient_byte_symbols = tuple(bytes([b]) for b in range(256))
        self._efficient_pretoken_cache: dict[bytes, tuple[int, ...]] = {}
        self._efficient_pretoken_cache_max_size = 100_000

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Self:
        gpt2_byte_decoder = {v: k for k, v in _gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath, encoding="utf-8") as vocab_f:
            gpt2_vocab: dict[str, int] = json.load(vocab_f)

        vocab = {
            token_id: bytes([gpt2_byte_decoder[ch] for ch in token_text])
            for token_text, token_id in gpt2_vocab.items()
        }

        if special_tokens:
            vocab_values = set(vocab.values())
            for special_token in special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in vocab_values:
                    vocab[len(vocab)] = special_token_bytes
                    vocab_values.add(special_token_bytes)

        gpt2_bpe_merges: list[tuple[str, str]] = []
        with open(merges_filepath, encoding="utf-8") as merges_f:
            for line in merges_f:
                cleaned_line = line.rstrip()
                parts = cleaned_line.split(" ")
                if cleaned_line and len(parts) == 2:
                    gpt2_bpe_merges.append((parts[0], parts[1]))

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_left]),
                bytes([gpt2_byte_decoder[token] for token in merge_right]),
            )
            for merge_left, merge_right in gpt2_bpe_merges
        ]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        parts = (
            [part for part in self._special_token_re.split(text) if part]
            if self._special_token_re is not None
            else [text]
        )

        pretokenized_parts: list[str] = []
        for part in parts:
            if part in self._special_token_set:
                pretokenized_parts.append(part)
                continue

            for match in self._pretok_re.finditer(part):
                pretokenized_parts.append(match.group(0))

        tokenized: list[int] = []

        for token in pretokenized_parts:
            token_bytes = token.encode("utf-8")

            if token in self.special_tokens:
                tokenized.append(self.bytes_to_id[token_bytes])

            else:
                token_bytes_list = [token_bytes[i:i + 1] for i in range(len(token_bytes))]

                while True:
                    pairs = list(zip(token_bytes_list[:-1], token_bytes_list[1:]))

                    merge = min(
                        (pair for pair in pairs if pair in self.merge_ranks),
                        key=lambda pair: self.merge_ranks[pair],
                        default=None,
                    )

                    if merge is None:
                        break

                    new_token_bytes_list = []
                    i = 0
                    while i < len(token_bytes_list):
                        if (
                            i + 1 < len(token_bytes_list)
                            and (token_bytes_list[i], token_bytes_list[i + 1]) == merge
                        ):
                            new_token_bytes_list.append(token_bytes_list[i] + token_bytes_list[i + 1])
                            i += 2
                        else:
                            new_token_bytes_list.append(token_bytes_list[i])
                            i += 1

                    token_bytes_list = new_token_bytes_list

                for token_bytes in token_bytes_list:
                    tokenized.append(self.bytes_to_id[token_bytes])

        return tokenized

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        max_special_len = max((len(tok) for tok in self.special_tokens), default=0)
        guard_len = max(0, max_special_len - 1)

        for chunk in iterable:
            buffer += chunk

            parts = (
                [p for p in self._special_token_re.split(buffer) if p]
                if self._special_token_re is not None
                else [buffer]
            )

            if not parts:
                continue

            for part in parts[:-1]:
                if part in self._special_token_set:
                    yield self.special_token_to_id[part]
                else:
                    yield from self.encode(part)

            tail = parts[-1]

            cut = max(0, len(tail) - guard_len) if guard_len > 0 else len(tail)
            stable_prefix = tail[:cut]
            guarded_suffix = tail[cut:]

            pretoks = [m.group(0) for m in self._pretok_re.finditer(stable_prefix)]
            emitted_text = "".join(pretoks[:-1])
            unresolved_last_pretok = pretoks[-1] if pretoks else ""

            if emitted_text:
                yield from self.encode(emitted_text)

            buffer = unresolved_last_pretok + guarded_suffix

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        all_bytes = b"".join(self.vocab[i] for i in ids)
        return all_bytes.decode("utf-8", errors="replace")

    def _efficient_merge_pretoken_bytes(self, token_bytes: bytes) -> tuple[bytes, ...]:
        if len(token_bytes) <= 1:
            return (self._efficient_byte_symbols[token_bytes[0]],) if token_bytes else ()

        symbols = [self._efficient_byte_symbols[b] for b in token_bytes]
        prev_idx = [i - 1 for i in range(len(symbols))]
        next_idx = [i + 1 for i in range(len(symbols))]
        next_idx[-1] = -1
        alive = [True] * len(symbols)

        heap: list[tuple[int, int]] = []
        for i in range(len(symbols) - 1):
            rank = self.merge_ranks.get((symbols[i], symbols[i + 1]))
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        while heap:
            rank, left = heapq.heappop(heap)

            if not alive[left]:
                continue

            right = next_idx[left]
            if right == -1 or not alive[right]:
                continue

            current_rank = self.merge_ranks.get((symbols[left], symbols[right]))
            if current_rank != rank:
                continue

            symbols[left] = symbols[left] + symbols[right]
            alive[right] = False

            right_next = next_idx[right]
            next_idx[left] = right_next
            if right_next != -1:
                prev_idx[right_next] = left

            left_prev = prev_idx[left]
            if left_prev != -1:
                left_rank = self.merge_ranks.get((symbols[left_prev], symbols[left]))
                if left_rank is not None:
                    heapq.heappush(heap, (left_rank, left_prev))

            if right_next != -1:
                right_rank = self.merge_ranks.get((symbols[left], symbols[right_next]))
                if right_rank is not None:
                    heapq.heappush(heap, (right_rank, left))

        merged_tokens: list[bytes] = []
        i = 0
        while i != -1:
            if alive[i]:
                merged_tokens.append(symbols[i])
            i = next_idx[i]

        return tuple(merged_tokens)

    def _efficient_encode_pretoken(self, token_bytes: bytes) -> tuple[int, ...]:
        cached = self._efficient_pretoken_cache.get(token_bytes)
        if cached is not None:
            return cached

        merged_tokens = self._efficient_merge_pretoken_bytes(token_bytes)
        encoded = tuple(self.bytes_to_id[token] for token in merged_tokens)

        if len(self._efficient_pretoken_cache) >= self._efficient_pretoken_cache_max_size:
            self._efficient_pretoken_cache.clear()
        self._efficient_pretoken_cache[token_bytes] = encoded

        return encoded

    def efficient_encode(self, text: str) -> list[int]:
        parts = (
            [part for part in self._special_token_re.split(text) if part]
            if self._special_token_re is not None
            else [text]
        )

        tokenized: list[int] = []

        for part in parts:
            if part in self._special_token_set:
                tokenized.append(self.special_token_to_id[part])
                continue

            for match in self._pretok_re.finditer(part):
                tokenized.extend(self._efficient_encode_pretoken(match.group(0).encode("utf-8")))

        return tokenized

    def efficient_encode_to_uint16(self, text: str) -> np.ndarray:
        if len(self.vocab) > np.iinfo(np.uint16).max + 1:
            raise ValueError("efficient_encode_to_uint16 requires vocab size <= 65536.")

        return np.asarray(self.efficient_encode(text), dtype=np.uint16)

    @staticmethod
    def _init_efficient_encode_worker(
        path: str,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str],
    ) -> None:
        fd = os.open(path, os.O_RDONLY)
        Tokenizer._efficient_shared_mmap = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        os.close(fd)
        Tokenizer._efficient_worker_tokenizer = Tokenizer(vocab, merges, special_tokens)

    @staticmethod
    def _efficient_encode_chunk(task: tuple[int, int]) -> np.ndarray:
        assert Tokenizer._efficient_shared_mmap is not None, "worker not initialized"
        assert Tokenizer._efficient_worker_tokenizer is not None, "worker not initialized"

        start, end = task
        chunk = Tokenizer._efficient_shared_mmap[start:end].decode("utf-8", errors="ignore")
        return Tokenizer._efficient_worker_tokenizer.efficient_encode_to_uint16(chunk)

    def efficient_encode_file_multiprocess(
        self,
        path: str | os.PathLike,
        num_processes: int | None = None,
        chunks_per_process: int = 4,
    ) -> np.ndarray:
        num_processes = num_processes or (os.cpu_count() or 1)

        if len(self.vocab) > np.iinfo(np.uint16).max + 1:
            raise ValueError("efficient_encode_file_multiprocess requires vocab size <= 65536.")

        if not self.special_tokens:
            with open(path, encoding="utf-8", errors="ignore") as f:
                return self.efficient_encode_to_uint16(f.read())

        with open(path, "rb") as f:
            split_token = self.special_tokens[0].encode("utf-8")
            desired_num_chunks = max(num_processes * chunks_per_process, 1)
            boundaries = find_chunk_boundaries(f, desired_num_chunks, split_token)

        tasks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        if not tasks:
            return np.empty(0, dtype=np.uint16)

        chunk_arrays: list[np.ndarray] = []

        if num_processes <= 1 or len(tasks) <= 1:
            fd = os.open(path, os.O_RDONLY)
            try:
                with mmap.mmap(fd, 0, access=mmap.ACCESS_READ) as mm:
                    for start, end in tasks:
                        chunk = mm[start:end].decode("utf-8", errors="ignore")
                        chunk_arrays.append(self.efficient_encode_to_uint16(chunk))
            finally:
                os.close(fd)
        else:
            with Pool(
                processes=num_processes,
                initializer=Tokenizer._init_efficient_encode_worker,
                initargs=(os.fspath(path), self.vocab, self.merges, self.special_tokens),
            ) as pool:
                for chunk_ids in pool.imap(Tokenizer._efficient_encode_chunk, tasks):
                    chunk_arrays.append(chunk_ids)

        return np.concatenate(chunk_arrays) if len(chunk_arrays) > 1 else chunk_arrays[0]
