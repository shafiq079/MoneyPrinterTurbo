"""Deterministic narration scene timeline construction.

This module deliberately has no provider dependencies.  It turns either an SRT
file or the original narration into the small, stable artifact consumed by
future material-matching stages.
"""

import json
import math
import os
import re
import tempfile
import unicodedata
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
TIMING_TOLERANCE = 0.001


def _is_unspaced_text(text: str) -> bool:
    alphanumeric = [character for character in text if character.isalnum()]
    if not alphanumeric:
        return False
    return all(
        "CJK" in unicodedata.name(character, "")
        or "HIRAGANA" in unicodedata.name(character, "")
        or "KATAKANA" in unicodedata.name(character, "")
        or "HANGUL" in unicodedata.name(character, "")
        for character in alphanumeric
    )


def _join_narration_text(left: str, right: str) -> str:
    """Join cue text without rewriting punctuation, case, or unspaced scripts."""
    separator = "" if _is_unspaced_text(left) and _is_unspaced_text(right) else " "
    return f"{left}{separator}{right}"


def _validate_duration_bounds(
    min_scene_duration: float | None, max_clip_duration: float
) -> None:
    if min_scene_duration is None:
        return
    if (
        isinstance(min_scene_duration, bool)
        or not isinstance(min_scene_duration, (int, float))
        or not math.isfinite(min_scene_duration)
        or min_scene_duration <= 0
    ):
        raise ValueError("min_scene_duration must be a finite positive number")
    if (
        isinstance(max_clip_duration, bool)
        or not isinstance(max_clip_duration, (int, float))
        or not math.isfinite(max_clip_duration)
        or max_clip_duration <= 0
    ):
        raise ValueError(
            "max_clip_duration must be a finite positive number when "
            "min_scene_duration is active"
        )
    if min_scene_duration > max_clip_duration:
        raise ValueError(
            "min_scene_duration must be less than or equal to max_clip_duration"
        )


def _merge_short_segments(
    segments: list[tuple[str, float, float]],
    min_scene_duration: float,
    max_clip_duration: float,
) -> list[tuple[str, float, float]]:
    """Coalesce short narration cues, including only approved internal pauses."""
    normalized = [(" ".join(text.split()), start, end) for text, start, end in segments]
    threshold = min(min_scene_duration / 4, 0.35)
    merged: list[tuple[str, float, float]] = []
    position = 0
    while position < len(normalized):
        text, start, end = normalized[position]
        if not text:
            merged.append((text, start, end))
            position += 1
            continue

        cursor = position + 1
        while end - start < min_scene_duration and cursor < len(normalized):
            next_position = cursor
            if not normalized[next_position][0]:
                hold_start, hold_end = normalized[next_position][1:]
                if (
                    next_position + 1 >= len(normalized)
                    or not normalized[next_position + 1][0]
                    or hold_end - hold_start - threshold > TIMING_TOLERANCE
                ):
                    break
                next_position += 1
            next_text, _, next_end = normalized[next_position]
            if next_end - start - max_clip_duration > TIMING_TOLERANCE:
                break
            text = _join_narration_text(text, next_text)
            end = next_end
            cursor = next_position + 1

        merged.append((text, start, end))
        position = cursor

    # Give only the final short cue a safe backward opportunity. Explicit leading
    # and trailing holds cannot qualify because both endpoints must be narration.
    final_narration_position = next(
        (index for index in range(len(merged) - 1, -1, -1) if merged[index][0]), None
    )
    if (
        final_narration_position is not None
        and final_narration_position > 0
        and merged[final_narration_position][2] - merged[final_narration_position][1]
        < min_scene_duration
    ):
        previous_position = final_narration_position - 1
        hold = None
        if not merged[previous_position][0]:
            hold = merged[previous_position]
            previous_position -= 1
        if previous_position >= 0 and merged[previous_position][0]:
            previous = merged[previous_position]
            hold_allowed = hold is None or (
                hold[2] - hold[1] - threshold <= TIMING_TOLERANCE
            )
            if (
                hold_allowed
                and merged[final_narration_position][2]
                - previous[1]
                - max_clip_duration
                <= TIMING_TOLERANCE
            ):
                merged[previous_position : final_narration_position + 1] = [
                    (
                        _join_narration_text(
                            previous[0], merged[final_narration_position][0]
                        ),
                        previous[1],
                        merged[final_narration_position][2],
                    )
                ]
    return merged


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
    # The presence of whitespace identifies a word-delimited language even
    # when there are fewer words than timed chunks.  Keep those words intact
    # and let the remaining chunks become holds; character splitting is only
    # appropriate for genuinely unspaced narration.
    units = words if re.search(r"\s", text) else list(text)
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
    if not math.isfinite(duration) or duration <= 0:
        return

    split_count = 1
    if math.isfinite(max_clip_duration) and max_clip_duration > 0:
        split_count = max(1, math.ceil(duration / max_clip_duration))
    chunks = _split_text(text, split_count) if text else [""] * split_count
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
    min_scene_duration: float | None = None,
) -> list[NarrationScene]:
    """Build scenes from valid subtitles, falling back to the narration text."""
    if not math.isfinite(audio_duration) or audio_duration <= 0:
        return []
    _validate_duration_bounds(min_scene_duration, max_clip_duration)

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

    if min_scene_duration is not None:
        timeline_segments: list[tuple[str, float, float]] = []
        cursor = 0.0
        for text, start, end in segments:
            gap = start - cursor
            if gap > TIMING_TOLERANCE:
                timeline_segments.append(("", cursor, start))
            elif abs(gap) <= TIMING_TOLERANCE:
                start = cursor
            timeline_segments.append((text, start, end))
            cursor = end
        trailing_gap = audio_duration - cursor
        if trailing_gap > TIMING_TOLERANCE:
            timeline_segments.append(("", cursor, audio_duration))
        elif timeline_segments and abs(trailing_gap) <= TIMING_TOLERANCE:
            text, start, _ = timeline_segments[-1]
            timeline_segments[-1] = (text, start, audio_duration)
        segments = _merge_short_segments(
            timeline_segments, min_scene_duration, max_clip_duration
        )

    scenes: list[NarrationScene] = []
    cursor = 0.0
    for text, start, end in segments:
        gap = start - cursor
        if gap > TIMING_TOLERANCE:
            _append_segment(scenes, "", cursor, start, max_clip_duration)
        elif abs(gap) <= TIMING_TOLERANCE:
            start = cursor
        _append_segment(scenes, text, start, end, max_clip_duration)
        cursor = end
    trailing_gap = audio_duration - cursor
    if trailing_gap > TIMING_TOLERANCE:
        _append_segment(scenes, "", cursor, audio_duration, max_clip_duration)
    elif scenes and abs(trailing_gap) <= TIMING_TOLERANCE:
        final = scenes[-1]
        scenes[-1] = final.model_copy(
            update={
                "end_time": audio_duration,
                "duration": audio_duration - final.start_time,
            }
        )
    if scenes:
        if scenes[0].start_time != 0 or scenes[-1].end_time != audio_duration:
            raise ValueError("scene timeline does not cover the complete audio")
        for expected_index, scene in enumerate(scenes, 1):
            if scene.index != expected_index:
                raise ValueError("scene indexes are not contiguous and one-based")
            if (
                not all(
                    math.isfinite(value)
                    for value in (scene.start_time, scene.end_time, scene.duration)
                )
                or scene.duration <= 0
                or abs(scene.duration - (scene.end_time - scene.start_time))
                > TIMING_TOLERANCE
                or (
                    expected_index > 1
                    and scene.start_time != scenes[expected_index - 2].end_time
                )
            ):
                raise ValueError("scene timeline timing is invalid")
    return scenes


def create_scene_timeline(
    task_dir: str,
    narration: str,
    audio_duration: float,
    subtitle_path: str = "",
    max_clip_duration: float = 5,
    min_scene_duration: float | None = None,
) -> str:
    """Create ``scenes.json`` in a task directory and return its path."""
    scenes = build_scenes(
        narration=narration,
        audio_duration=audio_duration,
        subtitle_path=subtitle_path,
        max_clip_duration=max_clip_duration,
        min_scene_duration=min_scene_duration,
    )
    os.makedirs(task_dir, exist_ok=True)
    scene_path = os.path.join(task_dir, "scenes.json")
    payload = [scene.model_dump(mode="json") for scene in scenes]
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=task_dir, suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, scene_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return scene_path
