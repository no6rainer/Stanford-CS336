from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute bytes/token for an input text or JSON file and a matching "
            "JSON file of token IDs. Prints only the compression ratio."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the original text or JSON file.",
    )
    parser.add_argument(
        "--ids",
        required=True,
        help="Path to the JSON file containing token IDs.",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "text", "json"],
        default="auto",
        help="Interpret the input as plain text or JSON. Defaults to auto.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimal places to print. Defaults to 4.",
    )
    return parser.parse_args()


def load_text_payload(input_path: Path, input_format: str) -> str | list[str]:
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

    raise ValueError("JSON input must be either a string or a list of strings.")


def load_token_ids(ids_path: Path) -> list[int] | list[list[int]]:
    with ids_path.open(encoding="utf-8") as f:
        payload: Any = json.load(f)

    if isinstance(payload, list) and all(isinstance(item, int) for item in payload):
        return payload

    if isinstance(payload, list) and all(
        isinstance(item, list) and all(isinstance(token_id, int) for token_id in item)
        for item in payload
    ):
        return payload

    raise ValueError("Token IDs JSON must be either list[int] or list[list[int]].")


def count_bytes(payload: str | list[str]) -> int:
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))

    return sum(len(document.encode("utf-8")) for document in payload)


def count_tokens(token_ids: list[int] | list[list[int]]) -> int:
    if token_ids and isinstance(token_ids[0], list):
        nested_ids = token_ids
        return sum(len(document_ids) for document_ids in nested_ids)

    flat_ids = token_ids
    return len(flat_ids)


def validate_shapes(
    payload: str | list[str],
    token_ids: list[int] | list[list[int]],
) -> None:
    payload_is_list = isinstance(payload, list)
    ids_are_nested = bool(token_ids) and isinstance(token_ids[0], list)

    if payload_is_list != ids_are_nested:
        raise ValueError(
            "Input and token IDs do not match: use a single string with list[int], "
            "or a list of strings with list[list[int]].",
        )


def main() -> None:
    args = parse_args()

    payload = load_text_payload(Path(args.input), args.input_format)
    token_ids = load_token_ids(Path(args.ids))
    validate_shapes(payload, token_ids)

    total_bytes = count_bytes(payload)
    total_tokens = count_tokens(token_ids)
    compression_ratio = total_bytes / total_tokens

    print(f"Compression ratio: {compression_ratio:.{args.precision}f}")


if __name__ == "__main__":
    main()
