from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE).split())


def exact_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def simhash64(text: str) -> int:
    tokens = normalize(text).split()[:256]
    features = tokens if len(tokens) < 3 else [" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    features = list(dict.fromkeys(features)) or [""]
    if len(features) >= 8:
        positions = [round(index * (len(features) - 1) / 7) for index in range(8)]
        sampled = [features[position] for position in positions]
    else:
        sampled = [features[index % len(features)] for index in range(8)]
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
    return fours | eights


class FingerprintIndex:
    def __init__(self, path: Path):
        self.exact: set[str] = set()
        self.values: list[int] = []
        self.bands: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(4)]
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                self.exact.add(record["exact_sha256"])
                value = int(record["simhash64"], 16)
                position = len(self.values)
                self.values.append(value)
                for band in range(4):
                    self.bands[band][(value >> (band * 16)) & 0xFFFF].append(position)

    def match(self, text: str, max_hamming: int = 3) -> dict:
        exact = exact_hash(text)
        if exact in self.exact:
            return {"contaminated": True, "reason": "exact", "distance": 0}
        value = simhash64(text)
        candidates = set()
        for band in range(4):
            candidates.update(self.bands[band].get((value >> (band * 16)) & 0xFFFF, []))
        minimum = min(((value ^ self.values[index]).bit_count() for index in candidates), default=64)
        return {
            "contaminated": minimum <= max_hamming,
            "reason": "near_duplicate" if minimum <= max_hamming else None,
            "distance": minimum,
        }
