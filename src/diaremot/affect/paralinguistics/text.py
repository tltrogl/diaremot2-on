"""Text-centric utilities for paralinguistic analysis."""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

import numpy as np

from .config import COMPREHENSIVE_FILLER_WORDS, VOWELS


@lru_cache(maxsize=256)
def enhanced_word_tokenization(text: str) -> tuple[str, ...]:
    """Tokenize text with heuristics suited for speech transcripts."""

    if not text:
        return ()

    words: list[str] = []
    current_word: list[str] = []

    for char in text.lower():
        if char.isalnum():
            current_word.append(char)
        elif char in ("'", "-") and current_word:
            current_word.append(char)
        else:
            if current_word:
                word = "".join(current_word)
                if len(word) > 1 or word in ("i", "a"):
                    words.append(word)
                current_word = []

    if current_word:
        word = "".join(current_word)
        if len(word) > 1 or word in ("i", "a"):
            words.append(word)

    return tuple(words)


def _count_occurrences(haystack: Iterable[str], needle: str) -> int:
    return int(np.sum(np.fromiter((token == needle for token in haystack), dtype=np.int32)))


def advanced_disfluency_detection(words: tuple[str, ...], raw_text: str) -> tuple[int, int, int]:
    """Return counts for filler words, repetitions, and false starts."""

    if not words:
        return 0, 0, 0

    words_array = np.array(words)
    text_lower = f" {raw_text.lower()} "

    filler_count = 0
    multi_word_fillers = [
        "you know",
        "i mean",
        "sort of",
        "kind of",
        "let me see",
        "let's see",
    ]
    for phrase in multi_word_fillers:
        filler_count += text_lower.count(f" {phrase} ")

    single_word_fillers = COMPREHENSIVE_FILLER_WORDS - {
        phrase for phrase in multi_word_fillers if " " in phrase
    }
    for filler in single_word_fillers:
        if " " not in filler:
            filler_count += _count_occurrences(words_array, filler)

    repetition_count = 0
    if len(words) > 1:
        exclude_from_repetition = {"the", "a", "an", "to", "of", "in", "on", "at", "by"}

        for idx in range(len(words) - 1):
            current_word = words[idx]
            next_word = words[idx + 1]

            if (
                current_word == next_word
                and len(current_word) > 2
                and current_word not in exclude_from_repetition
            ):
                repetition_count += 1

        for idx in range(len(words) - 3):
            if words[idx] == words[idx + 2] and len(words[idx]) > 3:
                middle_word = words[idx + 1]
                if middle_word in COMPREHENSIVE_FILLER_WORDS or len(middle_word) <= 2:
                    repetition_count += 1

    false_start_count = 0

    punctuation_patterns = [" - ", " -- ", "... "]
    for pattern in punctuation_patterns:
        false_start_count += text_lower.count(pattern)

    restart_patterns = [
        r" i - i ",
        r" we - we ",
        r" they - they ",
        r" you - you ",
        r" he - he ",
        r" she - she ",
        r" it - it ",
    ]

    for pattern in restart_patterns:
        matches = re.findall(pattern, text_lower)
        false_start_count += len(matches)

    if len(words) >= 4:
        for idx in range(len(words) - 3):
            if (
                idx + 3 < len(words)
                and len(words[idx]) > 2
                and len(words[idx + 2]) > 2
                and words[idx] != words[idx + 2]
                and words[idx + 1] in {"um", "uh", "er"}
            ):
                false_start_count += 1

    return int(filler_count), int(repetition_count), int(false_start_count)


@lru_cache(maxsize=512)
def enhanced_syllable_estimation(word: str) -> int:
    """Enhanced syllable estimation with improved rules."""

    if not word or len(word) < 1:
        return 1

    word_lower = word.lower().strip(".,!?;:")

    if len(word_lower) == 1:
        return 1

    vowel_runs = 0
    prev_was_vowel = False
    silent_e = word_lower.endswith("e") and len(word_lower) > 2

    for idx, char in enumerate(word_lower):
        is_vowel = char in VOWELS

        if is_vowel:
            if not prev_was_vowel:
                vowel_runs += 1
            elif idx < len(word_lower) - 1:
                if char + word_lower[idx - 1] in {
                    "ai",
                    "au",
                    "ea",
                    "ee",
                    "ei",
                    "ie",
                    "oa",
                    "oo",
                    "ou",
                    "ui",
                }:
                    pass
        prev_was_vowel = is_vowel

    if silent_e and vowel_runs > 1:
        vowel_runs -= 1

    syllable_endings = ["tion", "sion", "cial", "tial", "ious", "eous"]
    for ending in syllable_endings:
        if word_lower.endswith(ending):
            vowel_runs += 1
            break

    return max(1, vowel_runs)


def vectorized_syllable_count(words: tuple[str, ...]) -> int:
    """Enhanced vectorized syllable counting."""

    if not words:
        return 0

    return sum(enhanced_syllable_estimation(word) for word in words)


__all__ = [
    "enhanced_word_tokenization",
    "advanced_disfluency_detection",
    "enhanced_syllable_estimation",
    "vectorized_syllable_count",
]
