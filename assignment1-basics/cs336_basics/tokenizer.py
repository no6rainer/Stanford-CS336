from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
import regex as re


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
            # Capture the delimiter so special tokens are preserved in the split result.
            self._special_token_re = re.compile(f"({split_pattern})")
            self._special_token_set = set(self.special_tokens)
        else:
            self._special_token_re = None
            self._special_token_set = set()

        
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
                token_bytes_list = [token_bytes[i:i+1] for i in range(len(token_bytes))]

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
