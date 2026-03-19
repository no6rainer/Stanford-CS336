from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cs336_basics.tokenizer import Tokenizer

DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize a text file or JSON file with a trained tokenizer and write "
            "the integer IDs to a JSON output file."
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
        help="Path to the input text or JSON file to tokenize.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSON file. Existing files will be overwritten.",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "text", "json"],
        default="auto",
        help="Interpret the input as plain text or JSON. Defaults to auto.",
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
        "--indent",
        type=int,
        default=None,
        help="Optional indentation level for the output JSON file.",
    )
    return parser.parse_args()


def load_input(input_path: Path, input_format: str) -> str | list[str]:
    resolved_format = input_format
    if resolved_format == "auto":
        resolved_format = "json" if input_path.suffix.lower() == ".json" else "text"

    if resolved_format == "text":
        return input_path.read_text(encoding="utf-8")

    with input_path.open(encoding="utf-8") as f:
        payload: Any = json.load(f)

    if isinstance(payload, str):
        return payload

    if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        return payload

    raise ValueError(
        "JSON input must be either a single string or a list of strings.",
    )


def tokenize_payload(tokenizer: Tokenizer, payload: str | list[str]) -> list[int] | list[list[int]]:
    if isinstance(payload, str):
        return tokenizer.encode(payload)

    return [tokenizer.encode(document) for document in payload]


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
    payload = load_input(input_path, args.input_format)
    token_ids = tokenize_payload(tokenizer, payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(token_ids, f, indent=args.indent)

    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        num_documents = len(token_ids)
        num_tokens = sum(len(document_ids) for document_ids in token_ids)
        print(
            f"Tokenized {num_documents} documents into {num_tokens} tokens. "
            f"Saved to {output_path}.",
        )
    else:
        print(f"Tokenized {len(token_ids)} tokens. Saved to {output_path}.")


if __name__ == "__main__":
    main()
