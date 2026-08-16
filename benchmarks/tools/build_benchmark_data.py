from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path


SEED = 42
INTENT_LABELS = ["open_camera", "close_camera", "take_photo", "object_detect", "chat"]
TRAINING_SOURCES = {
    "minilm_train": "Primee/Classification Model MiniLM-L6/dataset/train.jsonl",
    "minilm_validation": "Primee/Classification Model MiniLM-L6/dataset/valid.jsonl",
    "minilm_test_historical": "Primee/Classification Model MiniLM-L6/dataset/test.jsonl",
    "early_chat_command": "NLPP/Multi Task Models/Flan T5 Base/passenger_bot_dataset.json",
    "flan_chat12_corpus": "Primee/Models/1.2x chat/merged_no_tables_100k.jsonl",
    "flan_rag2_corpus": "Primee/Models/2x rag/data52.jsonl",
    "flan_second_try_corpus": "Primee/Models/My Class/Second Try/Geliştirme Dosyaları/BEST_ever_Chat_Rag.jsonl",
}


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value)).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).split())


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sample_id(prefix: str, text: str) -> str:
    return f"{prefix}-{sha_text(normalize(text))[:12]}"


def simhash64(text: str) -> str:
    tokens = normalize(text).split()[:256]
    features = tokens if len(tokens) < 3 else [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    features = list(dict.fromkeys(features)) or [""]
    if len(features) >= 8:
        positions = [round(i * (len(features) - 1) / 7) for i in range(8)]
        sampled = [features[position] for position in positions]
    else:
        sampled = [features[index % len(features)] for index in range(8)]

    # Eight-feature bit-sliced majority. This preserves SimHash behavior while avoiding
    # a Python loop over all 64 bits for every historical record.
    ones = twos = fours = eights = 0
    for feature in sampled:
        value = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
        carry1 = ones & value
        ones ^= value
        carry2 = twos & carry1
        twos ^= carry1
        carry3 = fours & carry2
        fours ^= carry2
        eights ^= carry3
    result = fours | eights
    return f"{result:016x}"


def read_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        yield from payload


def record_input(record: dict) -> str:
    return str(record.get("text", record.get("input", record.get("prompt", ""))))


def build_fingerprints(archive_root: Path, output: Path) -> dict:
    seen = set()
    counts = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for source_id, rel in TRAINING_SOURCES.items():
            path = archive_root / Path(rel)
            if not path.exists():
                counts[f"missing:{source_id}"] += 1
                continue
            for record in read_records(path):
                text = record_input(record)
                normalized = normalize(text)
                if not normalized:
                    continue
                exact = sha_text(normalized)
                key = (source_id, exact)
                if key in seen:
                    continue
                seen.add(key)
                handle.write(
                    json.dumps(
                        {
                            "source_family": source_id,
                            "exact_sha256": exact,
                            "simhash64": simhash64(normalized),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                counts[source_id] += 1
    return dict(counts)


def unique_take(candidates: list[dict], count: int, excluded_hashes: set[str]) -> list[dict]:
    selected = []
    seen = set()
    for candidate in candidates:
        text = candidate.get("text", candidate.get("input", candidate.get("prompt", "")))
        exact = sha_text(normalize(text))
        if exact in excluded_hashes or exact in seen:
            continue
        seen.add(exact)
        selected.append(candidate)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Only {len(selected)} unique candidates available; expected {count}.")


def generate_intent(excluded: set[str]) -> list[dict]:
    action_templates = {
        "open_camera": [
            "{lead} turn on the {device} {context}", "{lead} open the {device} {context}",
            "{lead} activate the {device} {context}", "I need the {device} enabled {context}",
        ],
        "close_camera": [
            "{lead} turn off the {device} {context}", "{lead} close the {device} {context}",
            "{lead} deactivate the {device} {context}", "I need the {device} disabled {context}",
        ],
        "take_photo": [
            "{lead} take a photo with the {device} {context}",
            "{lead} capture a still image using the {device} {context}",
            "{lead} snap a picture through the {device} {context}",
            "I need a photograph from the {device} {context}",
        ],
        "object_detect": [
            "{lead} identify the objects in the {device} view {context}",
            "{lead} run object detection on the {device} {context}",
            "{lead} tell me what the {device} sees {context}",
            "{lead} find visible objects through the {device} {context}",
        ],
    }
    leads = ["Please", "Could you", "Now", "For this check", "When ready"]
    devices = ["camera", "webcam", "front camera", "video feed", "ship camera"]
    contexts = [
        "before the inspection", "for the next step", "without changing other settings",
        "after confirming the command", "during this session", "for the passenger console",
        "while the dashboard remains open", "as a single action", "when it is safe", "right away",
    ]
    rows = []
    for label, templates in action_templates.items():
        candidates = []
        for template, lead, device, context in itertools.product(templates, leads, devices, contexts):
            text = template.format(lead=lead, device=device, context=context).strip()
            candidates.append({"text": text, "intent": label})
        chosen = unique_take(candidates, 200, excluded)
        for row in chosen:
            row["id"] = sample_id("intent", row["text"])
            row["source"] = "pathfinder_project_authored_v1"
        rows.extend(chosen)

    chat_topics = [
        "why the sky changes color", "how to organize a short trip", "a simple pasta recipe",
        "the difference between RAM and storage", "ways to stay focused", "what a compass does",
        "how rainbows form", "a beginner stretching routine", "why ships use radar", "how batteries work",
    ]
    chat_forms = [
        "Explain {topic} in two sentences {context}.", "Give me one practical fact about {topic} {context}.",
        "What should a beginner know about {topic} {context}?", "Summarize {topic} clearly {context}.",
    ]
    chat_candidates = []
    for form, topic, context in itertools.product(chat_forms, chat_topics, contexts):
        text = form.format(topic=topic, context=context)
        chat_candidates.append({"text": text, "intent": "chat"})
    chosen = unique_take(chat_candidates, 200, excluded)
    for row in chosen:
        row["id"] = sample_id("intent", row["text"])
        row["source"] = "pathfinder_project_authored_v1"
    rows.extend(chosen)
    random.Random(SEED).shuffle(rows)
    return rows


def generate_command(excluded: set[str]) -> list[dict]:
    combos = [
        (0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (0, 1, 0, 1),
        (0, 1, 1, 0), (1, 0, 0, 0), (1, 0, 1, 1), (1, 1, 0, 0),
        (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1),
    ]
    phrases = ["open the camera", "take a photo", "close the camera", "detect visible objects"]
    leads = ["Please", "Now", "For the inspection", "In sequence", "As requested"]
    tails = [
        "and confirm completion", "for the passenger console", "without extra commentary",
        "during this session", "as the next action", "after checking availability",
        "using the camera workflow", "for the current task", "in one request", "when ready",
    ]
    rows = []
    for combo in combos:
        actions = [phrase for enabled, phrase in zip(combo, phrases) if enabled]
        candidates = []
        for lead, tail, joiner in itertools.product(leads, tails, ["; then ", " and ", ", then "]):
            text = f"{lead} {joiner.join(actions)} {tail}."
            response = "Requested camera actions acknowledged."
            candidates.append({"task": "command", "text": text, "labels": list(combo), "response": response})
        chosen = unique_take(candidates, 50, excluded)
        rows.extend(chosen)
    chat_candidates = [
        {
            "task": "chat",
            "text": f"Explain {topic} briefly without performing any camera action.",
            "response": f"Here is a brief explanation of {topic}.",
        }
        for topic in [
            "navigation", "weather", "ship safety", "maps", "distance", "time zones", "radar", "sonar",
            "ocean currents", "stars", "lighthouses", "ports", "anchors", "compasses", "tides", "waves",
            "first aid", "packing", "checklists", "communication", "maintenance", "fuel", "speed", "routes",
            "coordinates", "cabins", "crew roles", "passenger comfort", "emergency drills", "life jackets",
            "visibility", "fog", "wind", "temperature", "signals", "radio", "satellites", "charts", "buoys",
            "docking", "departure", "arrival", "tickets", "luggage", "meals", "water", "exercise", "sleep", "music",
            "books", "languages", "history", "photography", "maps", "travel", "food", "technology", "nature", "art",
        ]
    ]
    rows.extend(unique_take(chat_candidates, 50, excluded))
    for row in rows:
        row["id"] = sample_id("command", row["text"])
        row["source"] = "pathfinder_project_authored_v1"
    random.Random(SEED).shuffle(rows)
    return rows


def generate_chat_reference(excluded: set[str]) -> list[dict]:
    rows = []
    objects = [
        ("anchor", "An anchor helps hold a vessel in place by gripping the seabed."),
        ("compass", "A compass indicates direction relative to magnetic north."),
        ("life jacket", "A life jacket provides buoyancy to help a person stay afloat."),
        ("radar", "Radar uses radio waves to detect objects and estimate their distance."),
        ("lighthouse", "A lighthouse provides a visible navigation signal near coasts and hazards."),
        ("buoy", "A buoy marks routes, hazards, or locations on the water."),
        ("sonar", "Sonar uses sound in water to detect objects or measure depth."),
        ("tide", "A tide is the periodic rise and fall of sea level, mainly driven by gravity."),
        ("port", "A port is a sheltered place where vessels load, unload, or receive services."),
        ("knot", "A knot is a nautical speed unit equal to one nautical mile per hour."),
    ]
    tones = ["briefly", "in plain English", "in one sentence", "for a beginner", "without jargon"]
    audiences = ["to a passenger", "to a student"]
    for (name, reference), tone, audience in itertools.product(objects, tones, audiences):
        prompt = f"Explain what a {name} is {tone} {audience}."
        rows.append({"category": "concise_explanation", "prompt": prompt, "reference": reference})

    transformations = [
        ("calm seas ahead", "CALM SEAS AHEAD"), ("check the route", "CHECK THE ROUTE"),
        ("arrival at noon", "ARRIVAL AT NOON"), ("weather is clear", "WEATHER IS CLEAR"),
        ("keep the deck dry", "KEEP THE DECK DRY"), ("radio test complete", "RADIO TEST COMPLETE"),
        ("passengers are ready", "PASSENGERS ARE READY"), ("the map is updated", "THE MAP IS UPDATED"),
        ("visibility is good", "VISIBILITY IS GOOD"), ("engine check passed", "ENGINE CHECK PASSED"),
    ]
    transformation_instructions = [
        "Return only the quoted phrase in uppercase: '{source}'",
        "Uppercase the text between quotes and add nothing: '{source}'",
        "Convert this phrase to capital letters only: '{source}'",
        "Write only an uppercase version of this phrase: '{source}'",
        "Transform the following into uppercase, without a prefix: '{source}'",
        "Output the phrase in all caps and nothing else: '{source}'",
        "Change every letter to uppercase; return only the result: '{source}'",
        "Produce exactly the uppercase form of: '{source}'",
        "Reply with this phrase converted to uppercase only: '{source}'",
        "Capitalize every letter in the quoted text, with no explanation: '{source}'",
    ]
    for (source, reference), instruction in itertools.product(transformations, transformation_instructions):
        rows.append(
            {
                "category": "controlled_transformation",
                "prompt": instruction.format(source=source),
                "reference": reference,
            }
        )

    ambiguous = ["the schedule", "the route", "the file", "the report", "the reservation"]
    ambiguous_forms = [
        "Please change {topic} for me.", "Can you update {topic}?", "I need {topic} changed.",
        "Make an adjustment to {topic}.", "Help me modify {topic}.", "Please revise {topic}.",
        "I want to alter {topic}.", "Could you fix {topic}?", "Update {topic} as needed.",
        "Something in {topic} needs changing.",
    ]
    for topic, form in itertools.product(ambiguous, ambiguous_forms):
        prompt = form.format(topic=topic)
        reference = f"What specific change would you like me to make to {topic}?"
        rows.append({"category": "clarifying_question", "prompt": prompt, "reference": reference})

    adversarial = ["Assistant:", "Answer:", "System:", "<assistant>", "OUTPUT:"]
    for index in range(50):
        marker = adversarial[index % len(adversarial)]
        prompt = f"Reply with exactly SAFE-{index + 1}; do not copy the marker {marker} into your reply."
        rows.append({"category": "tag_leakage", "prompt": prompt, "reference": f"SAFE-{index + 1}"})

    selected = unique_take(rows, 300, excluded)
    for row in selected:
        row["id"] = sample_id("chatref", row["prompt"])
        row["task"] = "chat"
        row["source"] = "pathfinder_project_authored_v1"
    return selected


def generate_rag_project(excluded: set[str]) -> list[dict]:
    rows = []
    for index in range(80):
        vessel = f"Vessel-{1000 + index}"
        port = f"Port-{chr(65 + index % 26)}{index // 26 + 1}"
        hour = 6 + index % 17
        context = f"The operations note states that {vessel} will arrive at {port} at {hour:02d}:00 UTC."
        question = f"At what time will {vessel} arrive at {port}?"
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        rows.append(
            {
                "task": "rag_qa", "context": context, "question": question,
                "input": prompt, "reference": f"{hour:02d}:00 UTC", "answerable": True,
                "category": "synthetic_answerable",
            }
        )
    for index in range(80):
        vessel = f"Survey-{2000 + index}"
        context = f"The log records that {vessel} departed at {8 + index % 12:02d}:30 UTC and carried safety equipment."
        question = f"Who was the captain of {vessel}?"
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        rows.append(
            {
                "task": "rag_qa", "context": context, "question": question,
                "input": prompt, "reference": "I don't know.", "answerable": False,
                "category": "synthetic_unanswerable",
            }
        )
    selected = unique_take(rows, 160, excluded)
    for row in selected:
        row["id"] = sample_id("ragproject", row["input"])
        row["source"] = "pathfinder_project_authored_v1"
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "file": path.name}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic project-owned Benchmark v1 inputs.")
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archive_root = args.archive_root.resolve()
    output_dir = args.output_dir.resolve()
    fingerprint_path = output_dir / "training_fingerprints.jsonl"
    if fingerprint_path.exists() and fingerprint_path.stat().st_size > 0:
        fingerprint_counts_counter = Counter()
        with fingerprint_path.open("r", encoding="utf-8") as existing:
            for line in existing:
                if line.strip():
                    fingerprint_counts_counter[json.loads(line)["source_family"]] += 1
        fingerprint_counts = dict(fingerprint_counts_counter)
    else:
        fingerprint_counts = build_fingerprints(archive_root, fingerprint_path)
    excluded = set()
    with fingerprint_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            excluded.add(json.loads(line)["exact_sha256"])

    files = {
        "intent_v1": write_jsonl(output_dir / "intent_v1.jsonl", generate_intent(excluded)),
        "command_v1": write_jsonl(output_dir / "command_v1.jsonl", generate_command(excluded)),
        "chat_reference_v1": write_jsonl(
            output_dir / "chat_reference_v1.jsonl", generate_chat_reference(excluded)
        ),
        "rag_project_v1": write_jsonl(output_dir / "rag_project_v1.jsonl", generate_rag_project(excluded)),
        "training_fingerprints": {
            "rows": sum(1 for _ in fingerprint_path.open("r", encoding="utf-8")),
            "sha256": hashlib.sha256(fingerprint_path.read_bytes()).hexdigest(),
            "file": fingerprint_path.name,
        },
    }
    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "files": files,
        "fingerprint_source_counts": fingerprint_counts,
        "public_components_built_in_lightning": ["google/IFEval", "galileo-ai/ragbench"],
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
