"""Deterministic narration scene timeline construction.

This module deliberately has no provider dependencies.  It turns either an SRT
file or the original narration into the small, stable artifact consumed by
future material-matching stages.
"""

import json
import math
import os
import re
from pathlib import Path

from pydantic import BaseModel

from app.services import subtitle
from app.utils import utils


class NarrationScene(BaseModel):
    """One ordered interval of narration."""

    index: int
    start_time: float
    end_time: float
    duration: float
    text: str


_SRT_TIME_RANGE = re.compile(
    r"^\s*(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*"
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*$"
)


def _timestamp_seconds(parts: tuple[str, ...]) -> float:
    hours, minutes, seconds, fraction = (int(value) for value in parts)
    if minutes >= 60 or seconds >= 60:
        return math.nan
    return hours * 3600 + minutes * 60 + seconds + fraction / (10 ** len(parts[3]))


def _text_weight(text: str) -> int:
    """Return a language-independent approximation of spoken text length."""
    return max(len(re.sub(r"\s+", "", text)), 1)


def _canonical_text(text: str) -> str:
    """Normalize spoken content for complete-subtitle validation."""
    normalized = utils.normalize_script_for_subtitle_matching(text)
    return "".join(
        character.casefold() for character in normalized if character.isalnum()
    )


def _split_text(text: str, count: int) -> list[str]:
    """Split text into ordered, approximately equal pieces.

    Word boundaries are preferred when there are enough words.  Character
    boundaries provide the same behavior for Chinese and other unspaced text.
    """
    text = " ".join(text.split())
    if count <= 1 or not text:
        return [text] if text else []

    words = text.split()
    units = words if len(words) >= count else list(text.replace(" ", ""))
    chunks: list[str] = []
    start = 0
    for part in range(count):
        end = round((part + 1) * len(units) / count)
        chunk_units = units[start:end]
        chunks.append(" ".join(chunk_units) if units is words else "".join(chunk_units))
        start = end
    # Keep exactly ``count`` chunks even when a very short utterance spans a
    # long interval.  Empty chunks represent the remaining timed hold/silence;
    # dropping them would recreate an overlong scene or leave time uncovered.
    return chunks


def _append_segment(
    scenes: list[NarrationScene],
    text: str,
    start: float,
    end: float,
    max_clip_duration: float,
) -> None:
    text = " ".join(text.split())
    duration = end - start
    if not text or not math.isfinite(duration) or duration <= 0:
        return

    split_count = 1
    if math.isfinite(max_clip_duration) and max_clip_duration > 0:
        split_count = max(1, math.ceil(duration / max_clip_duration))
    chunks = _split_text(text, split_count)
    # Time is divided evenly rather than by chunk text length.  Natural word
    # boundaries can make text chunks quite uneven; weighting time by those
    # lengths could therefore exceed max_clip_duration even though split_count
    # was calculated correctly.
    cursor = start
    for position, chunk in enumerate(chunks):
        chunk_end = (
            end
            if position == len(chunks) - 1
            else start + duration * (position + 1) / len(chunks)
        )
        scenes.append(
            NarrationScene(
                index=len(scenes) + 1,
                start_time=cursor,
                end_time=chunk_end,
                duration=chunk_end - cursor,
                text=chunk,
            )
        )
        cursor = chunk_end


def _subtitle_segments(
    subtitle_path: str, audio_duration: float
) -> list[tuple[str, float, float]]:
    segments: list[tuple[str, float, float]] = []
    previous_end = 0.0
    for _, time_range, text in subtitle.file_to_subtitles(subtitle_path):
        match = _SRT_TIME_RANGE.fullmatch(time_range)
        if not match:
            continue
        start = _timestamp_seconds(match.groups()[:4])
        end = _timestamp_seconds(match.groups()[4:])
        if not all(math.isfinite(value) for value in (start, end)) or end <= start:
            continue
        start = max(start, previous_end, 0.0)
        end = min(end, audio_duration)
        if end <= start or not text.strip():
            continue
        segments.append((text, start, end))
        previous_end = end
    return segments


def build_scenes(
    narration: str,
    audio_duration: float,
    subtitle_path: str = "",
    max_clip_duration: float = 5,
) -> list[NarrationScene]:
    """Build scenes from valid subtitles, falling back to the narration text."""
    if not math.isfinite(audio_duration) or audio_duration <= 0:
        return []

    segments = _subtitle_segments(subtitle_path, audio_duration)
    narration_content = _canonical_text(narration)
    subtitle_content = _canonical_text(" ".join(text for text, _, _ in segments))
    # A partially parseable SRT must not silently discard the narration cues
    # represented by malformed or missing blocks.  Subtitle timing is trusted
    # only when its ordered spoken content accounts for the complete script.
    if narration_content and subtitle_content != narration_content:
        segments = []
    if not segments:
        normalized = utils.normalize_script_for_subtitle_matching(narration)
        texts = utils.split_string_by_punctuations(normalized)
        texts = [text for text in texts if text.strip()]
        weights = [_text_weight(text) for text in texts]
        total_weight = sum(weights)
        cursor = 0.0
        segments = []
        for position, (text, weight) in enumerate(zip(texts, weights)):
            end = (
                audio_duration
                if position == len(texts) - 1
                else cursor + audio_duration * weight / total_weight
            )
            segments.append((text, cursor, end))
            cursor = end

    scenes: list[NarrationScene] = []
    for text, start, end in segments:
        _append_segment(scenes, text, start, end, max_clip_duration)
    return scenes


def create_scene_timeline(
    task_dir: str,
    narration: str,
    audio_duration: float,
    subtitle_path: str = "",
    max_clip_duration: float = 5,
) -> str:
    """Create ``scenes.json`` in a task directory and return its path."""
    scenes = build_scenes(
        narration=narration,
        audio_duration=audio_duration,
        subtitle_path=subtitle_path,
        max_clip_duration=max_clip_duration,
    )
    os.makedirs(task_dir, exist_ok=True)
    scene_path = os.path.join(task_dir, "scenes.json")
    payload = [scene.model_dump(mode="json") for scene in scenes]
    Path(scene_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return scene_path
