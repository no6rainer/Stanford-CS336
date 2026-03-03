import json
from pathlib import Path

from cs336_basics import train_bpe

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


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "owt_train.txt"
    output_dir = project_root / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab, merges = train_bpe(input_path, 32_000, ["<|endoftext|>"])

    vocab_path = output_dir / "owt_vocab_32000.json"
    merges_path = output_dir / "owt_merges_32000.txt"

    byte_encoder = gpt2_bytes_to_unicode()
    serialized_vocab = {
        bytes_to_gpt2_string(token_bytes, byte_encoder): token_id
        for token_id, token_bytes in sorted(vocab.items(), key=lambda x: x[0])
    }
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(serialized_vocab, f, ensure_ascii=False, indent=2)

    with merges_path.open("w", encoding="utf-8") as f:
        for left, right in merges:
            left_s = bytes_to_gpt2_string(left, byte_encoder)
            right_s = bytes_to_gpt2_string(right, byte_encoder)
            f.write(f"{left_s} {right_s}\n")

    print(f"saved vocab to: {vocab_path}")
    print(f"saved merges to: {merges_path}")
