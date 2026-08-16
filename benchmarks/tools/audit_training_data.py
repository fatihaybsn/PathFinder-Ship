from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


DATASETS = {
    "minilm": {
        "splits": {
            "train": "Primee/Classification Model MiniLM-L6/dataset/train.jsonl",
            "validation": "Primee/Classification Model MiniLM-L6/dataset/valid.jsonl",
            "test": "Primee/Classification Model MiniLM-L6/dataset/test.jsonl",
        }
    },
    "flan_chat12": {
        "splits": {
            "corpus": "Primee/Models/1.2x chat/merged_no_tables_100k.jsonl",
            "validation": "Primee/Models/1.2x chat/validation_5K_merged.jsonl",
        },
        "reconstruct_train_by_multiset_subtraction": True,
    },
    "flan_rag2": {
        "splits": {
            "corpus": "Primee/Models/2x rag/data52.jsonl",
            "validation": "Primee/Models/2x rag/validation.jsonl",
        },
        "reconstruct_train_by_multiset_subtraction": True,
    },
    "flan_second_try": {
        "splits": {
            "corpus": "Primee/Models/My Class/Second Try/Geliştirme Dosyaları/BEST_ever_Chat_Rag.jsonl",
            "validation": "Primee/Models/My Class/Second Try/Geliştirme Dosyaları/val.jsonl",
        },
        "reconstruct_train_by_multiset_subtraction": True,
    },
}


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value)).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).split())


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        yield from payload


def keys(record: dict) -> tuple[str, str]:
    input_text = record.get("text", record.get("input", ""))
    output = record.get("output", record.get("response", record.get("labels", "")))
    task = record.get("task", record.get("intent", ""))
    input_key = stable_hash(normalize(input_text))
    pair_key = stable_hash("|".join((normalize(task), normalize(input_text), normalize(output))))
    return input_key, pair_key


def audit_file(path: Path) -> dict:
    inputs: Counter[str] = Counter()
    pairs: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    rows = 0
    for record in read_records(path):
        rows += 1
        input_key, pair_key = keys(record)
        inputs[input_key] += 1
        pairs[pair_key] += 1
        tasks[str(record.get("task", record.get("intent", "unknown")))] += 1
    return {
        "rows": rows,
        "tasks": dict(tasks),
        "exact_pair_duplicate_rows": sum(count - 1 for count in pairs.values() if count > 1),
        "input_duplicate_rows": sum(count - 1 for count in inputs.values() if count > 1),
        "_input_keys": set(inputs),
        "_pair_keys": set(pairs),
        "_input_counts": inputs,
        "_pair_counts": pairs,
    }


def public_view(result: dict) -> dict:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def subtract_validation(corpus: dict, validation: dict) -> dict:
    """Reconstruct the effective train multiset when val was selected from a saved full corpus."""
    pair_counts = corpus["_pair_counts"].copy()
    input_counts = corpus["_input_counts"].copy()
    pair_counts.subtract(validation["_pair_counts"])
    input_counts.subtract(validation["_input_counts"])
    pair_counts = Counter({key: count for key, count in pair_counts.items() if count > 0})
    input_counts = Counter({key: count for key, count in input_counts.items() if count > 0})
    rows = max(0, corpus["rows"] - validation["rows"])
    return {
        "rows": rows,
        "tasks": {"note": "task counts are not reconstructed because validation rows are stored without source indices"},
        "exact_pair_duplicate_rows": sum(count - 1 for count in pair_counts.values() if count > 1),
        "input_duplicate_rows": sum(count - 1 for count in input_counts.values() if count > 1),
        "_input_keys": set(input_counts),
        "_pair_keys": set(pair_counts),
        "_input_counts": input_counts,
        "_pair_counts": pair_counts,
        "reconstruction": "full corpus minus validation multiset",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit known training, validation and test splits.")
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    root = args.archive_root.resolve()
    report = {"schema_version": 1, "datasets": {}}
    markdown = ["# Training Data Leakage Audit", ""]
    for dataset_id, specification in DATASETS.items():
        splits = specification["splits"]
        computed = {}
        missing = []
        for split, rel in splits.items():
            path = root / Path(rel)
            if not path.exists():
                missing.append(rel)
                continue
            computed[split] = audit_file(path)
        if (
            specification.get("reconstruct_train_by_multiset_subtraction")
            and "corpus" in computed
            and "validation" in computed
        ):
            computed["train_reconstructed"] = subtract_validation(
                computed["corpus"], computed["validation"]
            )
        overlaps = {}
        split_names = list(computed)
        for left_index, left in enumerate(split_names):
            for right in split_names[left_index + 1 :]:
                overlaps[f"{left}__{right}"] = {
                    "exact_pair_overlap": len(computed[left]["_pair_keys"] & computed[right]["_pair_keys"]),
                    "input_overlap": len(computed[left]["_input_keys"] & computed[right]["_input_keys"]),
                }
        report["datasets"][dataset_id] = {
            "splits": {name: public_view(value) for name, value in computed.items()},
            "overlaps": overlaps,
            "missing": missing,
            "method": (
                "effective train reconstructed by subtracting the validation multiset from the saved full corpus"
                if specification.get("reconstruct_train_by_multiset_subtraction")
                else "explicit saved splits"
            ),
        }
        markdown.extend([f"## `{dataset_id}`", ""])
        for split, value in computed.items():
            markdown.append(
                f"- {split}: {value['rows']:,} rows; {value['exact_pair_duplicate_rows']:,} exact-pair "
                f"duplicate rows; {value['input_duplicate_rows']:,} input duplicate rows."
            )
        for pair, value in overlaps.items():
            markdown.append(
                f"- {pair}: {value['exact_pair_overlap']:,} exact pair overlaps; "
                f"{value['input_overlap']:,} input overlaps."
            )
        if missing:
            markdown.append(f"- Missing: {', '.join(missing)}")
        markdown.append("")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_markdown.write_text("\n".join(markdown), encoding="utf-8")
    print(f"[audit] wrote {args.output_json} and {args.output_markdown}")


if __name__ == "__main__":
    main()
