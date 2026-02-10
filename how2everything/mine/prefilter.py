from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_ngrams(text: str, n: int) -> List[str]:
    words = _normalize_text(text).split()
    if len(words) < n:
        return []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _ngram_repetition_rate(steps: List[str], n: int) -> float:
    if not steps or len(steps) < 2:
        return 0.0
    all_ngrams: List[str] = []
    for step in steps:
        all_ngrams.extend(_extract_ngrams(step, n))
    if not all_ngrams:
        return 0.0
    counts = Counter(all_ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(all_ngrams)


def is_repetitive_steps(steps: List[str]) -> bool:
    return (
        _ngram_repetition_rate(steps, 4) >= 0.3
        or _ngram_repetition_rate(steps, 3) >= 0.35
        or _ngram_repetition_rate(steps, 2) >= 0.4
    )


def check_step_count(steps: List[str], min_steps: int, max_steps: int) -> Tuple[bool, str]:
    n = len(steps)
    if n < min_steps:
        return False, f"Too few steps: {n} < {min_steps}"
    if n > max_steps:
        return False, f"Too many steps: {n} > {max_steps}"
    return True, ""


def apply_prefilter(steps: List[str], min_steps: int, max_steps: int) -> Tuple[bool, str]:
    ok, reason = check_step_count(steps, min_steps, max_steps)
    if not ok:
        return False, reason
    if is_repetitive_steps(steps):
        return False, "High repetition rate"
    return True, ""

