from pathlib import Path
import json
import random

SEP = "<|endoftext|>"


def sample_documents(path: str | Path, k: int = 10, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    samples: list[str] = []
    seen = 0
    buffer = ""

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), ""):
            buffer += chunk
            parts = buffer.split(SEP)
            buffer = parts.pop()

            for doc in parts:
                doc = doc.strip()
                if not doc:
                    continue
                seen += 1
                if len(samples) < k:
                    samples.append(doc)
                else:
                    j = rng.randrange(seen)
                    if j < k:
                        samples[j] = doc

    last_doc = buffer.strip()
    if last_doc:
        seen += 1
        if len(samples) < k:
            samples.append(last_doc)
        else:
            j = rng.randrange(seen)
            if j < k:
                samples[j] = last_doc

    return samples


def save_samples() -> None:
    tiny_docs = sample_documents("data/TinyStoriesV2-GPT4-train.txt", k=10, seed=42)
    owt_docs = sample_documents("data/owt_train.txt", k=10, seed=42)

    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/tinystories_sample_10.json", "w", encoding="utf-8") as f:
        json.dump(tiny_docs, f, ensure_ascii=False, indent=2)

    with open("artifacts/owt_sample_10.json", "w", encoding="utf-8") as f:
        json.dump(owt_docs, f, ensure_ascii=False, indent=2)

    print("TinyStories sampled:", len(tiny_docs))
    print("OpenWebText sampled:", len(owt_docs))


if __name__ == "__main__":
    save_samples()
