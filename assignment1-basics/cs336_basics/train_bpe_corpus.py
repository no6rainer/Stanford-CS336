from __future__ import annotations

import argparse
import json
from pathlib import Path

from cs336_basics import train_bpe

DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]


def gpt2_bytes_to_unicode() -> dict[int, str]:
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
    return dict(zip(bs, map(chr, cs), strict=True))


def bytes_to_gpt2_string(token: bytes, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[b] for b in token)


def train_and_save_bpe(
    input_path: str | Path,
    vocab_size: int,
    vocab_output_path: str | Path,
    merges_output_path: str | Path,
    special_tokens: list[str] | None = None,
) -> None:
    special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS

    vocab_output_path = Path(vocab_output_path)
    merges_output_path = Path(merges_output_path)
    vocab_output_path.parent.mkdir(parents=True, exist_ok=True)
    merges_output_path.parent.mkdir(parents=True, exist_ok=True)

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    byte_encoder = gpt2_bytes_to_unicode()
    serialized_vocab = {
        bytes_to_gpt2_string(token_bytes, byte_encoder): token_id
        for token_id, token_bytes in sorted(vocab.items(), key=lambda x: x[0])
    }

    with vocab_output_path.open("w", encoding="utf-8") as f:
        json.dump(serialized_vocab, f, ensure_ascii=False, indent=2)

    with merges_output_path.open("w", encoding="utf-8") as f:
        for left, right in merges:
            left_s = bytes_to_gpt2_string(left, byte_encoder)
            right_s = bytes_to_gpt2_string(right, byte_encoder)
            f.write(f"{left_s} {right_s}\n")

    print(f"saved vocab to: {vocab_output_path}")
    print(f"saved merges to: {merges_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a byte-level BPE tokenizer on a plaintext corpus and save "
            "the resulting vocab and merges files."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the plaintext corpus to train on.",
    )
    parser.add_argument(
        "--vocab-size",
        required=True,
        type=int,
        help="Target vocabulary size, including special tokens.",
    )
    parser.add_argument(
        "--output-vocab",
        required=True,
        help="Path to the output vocab JSON file.",
    )
    parser.add_argument(
        "--output-merges",
        required=True,
        help="Path to the output merges TXT file.",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        help=(
            "Special token to include in the vocabulary. "
            "Repeat the flag to add multiple special tokens. "
            "Defaults to <|endoftext|>."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_and_save_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        vocab_output_path=args.output_vocab,
        merges_output_path=args.output_merges,
        special_tokens=args.special_tokens,
    )


if __name__ == "__main__":
    main()
