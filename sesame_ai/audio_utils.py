# sesame_ai/audio_utils.py

import numpy as np


def compress_silence(
    audio_i16: np.ndarray,
    sample_rate: int,
    *,
    silence_threshold: float = 200.0,
    max_pause_ms: int = 200,
    frame_ms: int = 10,
) -> np.ndarray:
    """
    Truncate silence gaps in PCM16 audio so they don't trigger server-side VAD.

    Silences longer than *max_pause_ms* are shortened to *max_pause_ms*.
    Shorter gaps pass through unchanged, preserving natural micro-pauses.
    Trailing silence is kept.
    """
    if audio_i16.size == 0:
        return audio_i16

    frame_samples = max(1, int(sample_rate * frame_ms / 1000))
    max_keep = max(1, int(sample_rate * max_pause_ms / 1000))

    num_frames = len(audio_i16) // frame_samples
    if num_frames < 2:
        return audio_i16

    trimmed_len = num_frames * frame_samples
    main = audio_i16[:trimmed_len].reshape(num_frames, frame_samples)
    trailing = audio_i16[trimmed_len:]

    rms = np.sqrt(np.mean(main.astype(np.float64) ** 2, axis=1))
    is_silence = rms < silence_threshold

    result_frames = []
    silence_run = []

    for i in range(num_frames):
        if is_silence[i]:
            silence_run.append(main[i])
        else:
            _flush_silence(result_frames, silence_run, max_keep)
            silence_run = []
            result_frames.append(main[i])

    if silence_run:
        result_frames.append(np.concatenate(silence_run))

    if not result_frames:
        return audio_i16

    result = np.concatenate(result_frames)
    if trailing.size:
        result = np.concatenate([result, trailing])
    return result


def _flush_silence(result_frames, silence_run, max_keep):
    if not silence_run:
        return
    total = np.concatenate(silence_run)
    if len(total) > max_keep:
        total = total[:max_keep]
    result_frames.append(total)


def extract_longest_utterance(
    audio_i16: np.ndarray,
    sample_rate: int,
    *,
    min_split_silence: float = 2.0,
    silence_threshold: float = 200.0,
    frame_ms: int = 20,
) -> np.ndarray:
    """Return the best utterance segment from *audio_i16*.

    The audio is split into segments wherever a silence gap exceeds
    *min_split_silence* seconds (default 2.0 — high enough to avoid
    splitting mid-sentence pauses).  Segments are then scored by
    ``duration * position_weight``:

    ============  ======  ============================
    Position      Weight  Rationale
    ============  ======  ============================
    first         0.0     Always the greeting — discard
    middle        1.0     The actual reply content
    last          0.5     Penalised (likely follow-up)
    ============  ======  ============================

    The highest-scoring segment is returned.  Single-segment audio
    (the segment is both first and last) is returned as-is.
    If no segments are found the original audio is returned unchanged.
    """
    if audio_i16.size == 0:
        return audio_i16

    frame_samples = max(1, int(sample_rate * frame_ms / 1000))
    min_silence_frames = max(
        1, int(sample_rate * min_split_silence / frame_samples)
    )

    num_frames = len(audio_i16) // frame_samples
    if num_frames < 2:
        return audio_i16

    trimmed_len = num_frames * frame_samples
    frames = audio_i16[:trimmed_len].reshape(num_frames, frame_samples)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    is_silence = rms < silence_threshold

    # Find runs of silence frames.
    silence_starts = []
    silence_ends = []
    in_silence = False
    for i in range(num_frames):
        if is_silence[i] and not in_silence:
            silence_starts.append(i)
            in_silence = True
        elif not is_silence[i] and in_silence:
            silence_ends.append(i)
            in_silence = False
    if in_silence:
        silence_ends.append(num_frames)

    # Find silence gaps long enough to split utterances.
    split_indices = [0]
    for s, e in zip(silence_starts, silence_ends):
        if e - s >= min_silence_frames:
            split_indices.append(s)
            split_indices.append(e)
    split_indices.append(num_frames)

    # Build segments and score by position + duration.
    num_segments = (len(split_indices) - 1) // 2
    if num_segments == 0:
        return audio_i16

    best_segment = audio_i16
    best_score = -1.0
    for seg_idx in range(num_segments):
        i = seg_idx * 2
        start_frame = split_indices[i]
        end_frame = split_indices[i + 1]
        seg_duration = (end_frame - start_frame) * frame_ms / 1000.0

        # Position-based weight
        if num_segments == 1:
            weight = 1.0
        elif seg_idx == 0:
            weight = 0.0   # first = greeting
        elif seg_idx == num_segments - 1:
            weight = 0.5   # last = follow-up penalty
        else:
            weight = 1.0   # middle = reply

        score = seg_duration * weight
        if score > best_score:
            best_score = score
            best_segment = frames[start_frame:end_frame].ravel()

    # Append any trailing samples that didn't fill a full frame.
    if best_score >= 0 and trimmed_len < len(audio_i16):
        best_segment = np.concatenate([best_segment, audio_i16[trimmed_len:]])

    return best_segment.astype(np.int16)
