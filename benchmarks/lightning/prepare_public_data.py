from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

from benchmarks.decontam import FingerprintIndex, exact_hash


RAGBENCH_CONFIGS = [
    "covidqa", "cuad", "delucionqa", "emanual", "expertqa", "finqa",
    "hagrid", "msmarco", "pubmedqa", "tatqa", "techqa",
]


def write_jsonl(path: Path, records: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"file": path.name, "rows": len(records), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare_ifeval(index: FingerprintIndex, output_dir: Path) -> dict:
    dataset = load_dataset("google/IFEval", split="train")
    clean = []
    rejected = []
    for record in dataset:
        prompt = record["prompt"]
        decision = index.match(prompt)
        target = rejected if decision["contaminated"] else clean
        target.append(
            {
                "id": f"ifeval-{record['key']}",
                "key": record["key"],
                "prompt": prompt,
                "instruction_id_list": record["instruction_id_list"],
                "kwargs": record["kwargs"],
                "source": "google/IFEval",
                "decontamination": decision,
            }
        )
    info = write_jsonl(output_dir / "ifeval_v1.jsonl", clean)
    write_jsonl(output_dir / "ifeval_rejected.jsonl", rejected)
    info["rejected"] = len(rejected)
    return info


def prepare_ragbench(index: FingerprintIndex, output_dir: Path) -> dict:
    rng = random.Random(42)
    selected = []
    rejected = []
    per_config = {}
    for config in RAGBENCH_CONFIGS:
        dataset = list(load_dataset("galileo-ai/ragbench", config, split="test"))
        rng.shuffle(dataset)
        accepted = 0
        for record in dataset:
            context = "\n\n".join(record.get("documents") or [])
            question = record.get("question", "")
            model_input = f"Context: {context}\nQuestion: {question}\nAnswer:"
            decision = index.match(model_input)
            prepared = {
                "id": f"ragbench-{config}-{record['id']}",
                "task": "rag_qa",
                "context": context,
                "question": question,
                "input": model_input,
                "reference": record.get("response", ""),
                "answerable": True,
                "source": f"galileo-ai/ragbench:{config}:test",
                "decontamination": decision,
            }
            if decision["contaminated"]:
                rejected.append(prepared)
                continue
            if accepted < 40:
                selected.append(prepared)
                accepted += 1
        if accepted != 40:
            raise RuntimeError(f"RAGBench {config}: only {accepted} clean examples; 40 required")
        per_config[config] = accepted
    info = write_jsonl(output_dir / "ragbench_v1.jsonl", selected)
    write_jsonl(output_dir / "ragbench_rejected.jsonl", rejected)
    info.update({"rejected": len(rejected), "per_config": per_config})
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    index = FingerprintIndex(args.fingerprints)
    manifest = {
        "schema_version": 1,
        "decontamination": {"exact": "normalized SHA-256", "near_duplicate": "sampled SimHash64, Hamming <= 3"},
        "ifeval": prepare_ifeval(index, args.output_dir),
        "ragbench": prepare_ragbench(index, args.output_dir),
    }
    (args.output_dir / "public_dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
