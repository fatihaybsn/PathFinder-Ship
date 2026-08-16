from __future__ import annotations

import math
import random
import re
import string
from collections import Counter
from statistics import mean
from typing import Callable, Iterable, Sequence


def normalize_answer(text: str) -> str:
    lowered = text.casefold()
    without_articles = re.sub(r"\b(a|an|the)\b", " ", lowered)
    without_punctuation = "".join(character for character in without_articles if character not in string.punctuation)
    return " ".join(without_punctuation.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    previous = [0] * (len(expected) + 1)
    for predicted_token in predicted:
        current = [0]
        for column, expected_token in enumerate(expected, 1):
            if predicted_token == expected_token:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(predicted)
    recall = lcs / len(expected)
    return 0.0 if lcs == 0 else 2 * precision * recall / (precision + recall)


def no_answer_match(prediction: str) -> float:
    normalized = normalize_answer(prediction)
    accepted = {"i dont know", "not in context", "answer is not in context", "unknown"}
    return float(normalized in accepted or normalized.startswith("i dont know"))


def lexical_context_support(prediction: str, context: str) -> float:
    predicted = [token for token in normalize_answer(prediction).split() if len(token) > 2]
    if not predicted:
        return 0.0
    context_tokens = set(normalize_answer(context).split())
    return sum(token in context_tokens for token in predicted) / len(predicted)


def repetition_rate(text: str, ngram_size: int = 3) -> float:
    tokens = normalize_answer(text).split()
    ngrams = [tuple(tokens[index : index + ngram_size]) for index in range(max(0, len(tokens) - ngram_size + 1))]
    if not ngrams:
        return 0.0
    return 1 - len(set(ngrams)) / len(ngrams)


def tag_leak(text: str) -> float:
    return float(bool(re.search(r"(?i)(?:^|\s)(assistant:|answer:|system:|<assistant>|output:)", text)))


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("confidence and correctness arrays must be non-empty and equal length")
    total = len(confidences)
    result = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence < upper or (bin_index == bins - 1 and confidence == 1.0)
        ]
        if not members:
            continue
        accuracy = mean(float(correct[index]) for index in members)
        confidence = mean(confidences[index] for index in members)
        result += len(members) / total * abs(accuracy - confidence)
    return result


def classification_summary(labels: Sequence[str], predictions: Sequence[str]) -> dict:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels and predictions must be non-empty and equal length")
    classes = sorted(set(labels) | set(predictions))
    per_class = {}
    supports = Counter(labels)
    for label in classes:
        tp = sum(expected == label and predicted == label for expected, predicted in zip(labels, predictions))
        fp = sum(expected != label and predicted == label for expected, predicted in zip(labels, predictions))
        fn = sum(expected == label and predicted != label for expected, predicted in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": supports[label]}
    accuracy = mean(expected == predicted for expected, predicted in zip(labels, predictions))
    macro_f1 = mean(value["f1"] for value in per_class.values())
    weighted_f1 = sum(value["f1"] * value["support"] for value in per_class.values()) / len(labels)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "per_class": per_class}


def bootstrap_ci(
    values: Sequence[object],
    metric: Callable[[Sequence[object]], float],
    iterations: int = 1000,
    seed: int = 42,
) -> dict:
    if not values:
        raise ValueError("cannot bootstrap an empty sequence")
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(metric(sample))
    estimates.sort()
    lower = estimates[math.floor(0.025 * (iterations - 1))]
    upper = estimates[math.ceil(0.975 * (iterations - 1))]
    return {"estimate": metric(values), "ci95_low": lower, "ci95_high": upper, "iterations": iterations}


def mean_metric(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else float("nan")
