from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer

DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize a plaintext corpus with the efficient multiprocess tokenizer "
            "and save token IDs as a NumPy uint16 array."
        ),
    )
    parser.add_argument(
        "--vocab",
        required=True,
        help="Path to the tokenizer vocab JSON file.",
    )
    parser.add_argument(
        "--merges",
        required=True,
        help="Path to the tokenizer merges TXT file.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the plaintext corpus to tokenize.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output .npy file. Existing files will be overwritten.",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        help=(
            "Special token to register with the tokenizer. "
            "Repeat the flag to add multiple special tokens. "
            "Defaults to <|endoftext|>."
        ),
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes to use. Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--chunks-per-process",
        type=int,
        default=4,
        help="Number of chunks to assign per process when splitting the corpus.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vocab_path = Path(args.vocab)
    merges_path = Path(args.merges)
    input_path = Path(args.input)
    output_path = Path(args.output)
    special_tokens = args.special_tokens or DEFAULT_SPECIAL_TOKENS

    tokenizer = Tokenizer.from_files(
        str(vocab_path),
        str(merges_path),
        special_tokens,
    )

    token_ids = tokenizer.efficient_encode_file_multiprocess(
        input_path,
        num_processes=args.num_processes,
        chunks_per_process=args.chunks_per_process,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        np.save(f, token_ids, allow_pickle=False)

    print(
        f"Tokenized {input_path} into {int(token_ids.size)} uint16 token IDs. "
        f"Saved to {output_path}.",
    )


if __name__ == "__main__":
    main()
