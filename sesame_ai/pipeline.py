# sesame_ai/pipeline.py — Core TTS → WebSocket → record pipeline.

import os
import time
import wave
import math
import tempfile
import logging
import subprocess
import shutil

import numpy as np

from .websocket import SesameWebSocket
from .audio_utils import compress_silence

logger = logging.getLogger("sesame.pipeline")

SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# audio I/O helpers
# ---------------------------------------------------------------------------

def _read_wav_mono_pcm16(path: str):
    with wave.open(path, "rb") as wf:
        sampwidth = wf.getsampwidth()
        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit WAV, got sampwidth={sampwidth}")
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        if wf.getnchannels() == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return rate, audio


def write_wav(path: str, pcm: bytes, rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(rate))
        wf.writeframes(pcm)


def wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "96k") -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", wav_path, "-vn", "-codec:a", "libmp3lame",
             "-b:a", bitrate, mp3_path], check=True)
        return
    lame = shutil.which("lame")
    if lame:
        br = bitrate.lower().replace("k", "")
        subprocess.run(["lame", "--silent", "-b", br, wav_path, mp3_path], check=True)
        return
    raise RuntimeError("No MP3 encoder found. Install ffmpeg or lame.")


# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------

def _resample(audio_i16: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio_i16.size == 0:
        return audio_i16
    duration = audio_i16.size / float(src_rate)
    dst_len = max(1, int(math.ceil(duration * dst_rate)))
    x_old = np.linspace(0.0, duration, num=audio_i16.size, endpoint=False)
    x_new = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, audio_i16.astype(np.float32)).astype(np.int16)


def _normalize(audio_i16: np.ndarray, target_peak: int = 14000) -> np.ndarray:
    if audio_i16.size == 0:
        return audio_i16
    peak = int(np.max(np.abs(audio_i16)))
    if peak <= 0:
        return audio_i16
    return np.clip(audio_i16.astype(np.float32) * (target_peak / peak),
                   -32768, 32767).astype(np.int16)


def _rms(audio_bytes: bytes) -> float:
    if not audio_bytes:
        return 0.0
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def _synthesize_tts(text: str, wav_path: str) -> None:
    """Synthesize text to a 16-bit mono WAV file.

    On Windows, pyttsx3 (SAPI5) is tried first.  Falls back to gTTS
    (requires internet + ffmpeg).
    """
    # 1 — pyttsx3 (Windows SAPI5, offline)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        return
    except Exception:
        pass

    # 2 — gTTS (Google TTS, online)
    try:
        from gtts import gTTS
    except ImportError:
        raise RuntimeError(
            "No TTS engine available. Install pyttsx3 (Windows) or gTTS (cross-platform)."
        )

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required when using gTTS (MP3→WAV conversion).")

    with tempfile.TemporaryDirectory() as td:
        mp3_path = os.path.join(td, "tts.mp3")
        tts = gTTS(text=text, lang="en")
        tts.save(mp3_path)
        # Convert MP3 → 16kHz mono WAV
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", mp3_path, "-vn", "-ar", str(SAMPLE_RATE),
             "-ac", "1", "-sample_fmt", "s16", wav_path], check=True)


# ---------------------------------------------------------------------------
# WebSocket streaming helpers
# ---------------------------------------------------------------------------

def _stream_pcm(ws: SesameWebSocket, pcm_bytes: bytes, sample_rate: int,
                chunk_ms: int = 20) -> None:
    samples_per_chunk = int(sample_rate * (chunk_ms / 1000.0))
    step = max(1, samples_per_chunk) * 2
    for i in range(0, len(pcm_bytes), step):
        ws.send_audio_data(pcm_bytes[i:i + step])
        time.sleep(chunk_ms / 1000.0)


def _send_silence(ws: SesameWebSocket, seconds: float, sample_rate: int) -> None:
    samples = int(seconds * sample_rate)
    if samples <= 0:
        return
    _stream_pcm(ws, np.zeros(samples, dtype=np.int16).tobytes(), sample_rate)


# ---------------------------------------------------------------------------
# drain & record
# ---------------------------------------------------------------------------

def _drain_until_silence(ws: SesameWebSocket, *,
                         max_wait: float = 30.0,
                         silence_seconds: float = 2.0,
                         label: str = "drain") -> int:
    """Discard audio chunks until silence indicates end of utterance."""
    start = time.time()
    last_audio: float | None = None
    last_ping = start
    discarded = 0

    logger.info("[%s] waiting for greeting to end ...", label)

    while True:
        if time.time() - start > max_wait:
            logger.info("[%s] max wait reached, %d chunks discarded", label, discarded)
            break
        if not ws.is_connected():
            logger.warning("[%s] disconnected during drain", label)
            break
        if (time.time() - last_ping) >= 10.0:
            ws.ping()
            last_ping = time.time()

        chunk = ws.get_next_audio_chunk(timeout=0.5)
        if chunk:
            discarded += 1
            if last_audio is None:
                logger.info("[%s] first audio at +%.1fs", label, time.time() - start)
            last_audio = time.time()
        elif last_audio is not None:
            if (time.time() - last_audio) >= silence_seconds:
                logger.info("[%s] silence reached, %d chunks discarded", label, discarded)
                break

    return discarded


def _record_until_silence(ws: SesameWebSocket, *,
                          max_duration: float = 180.0,
                          silence_seconds: float = 3.0,
                          noise_floor: int = 200) -> bytes:
    """Record audio chunks until silence or max duration.

    Only chunks whose RMS exceeds *noise_floor* count as speech.
    Server-side comfort noise (sub-200 RMS) resets nothing, so the
    silence timer can actually fire.
    """
    start = time.time()
    last_speech: float | None = None
    last_ping = start
    chunks: list[bytes] = []
    was_connected = True

    logger.info("[record] listening ...")

    while True:
        now = time.time()
        if now - start > max_duration:
            logger.info("[record] max duration reached")
            break
        if not ws.is_connected():
            logger.warning("[record] server disconnected — %d chunks captured", len(chunks))
            was_connected = False
            break
        if (now - last_ping) >= 10.0:
            ws.ping()
            last_ping = now

        chunk = ws.get_next_audio_chunk(timeout=0.5)
        if chunk:
            chunks.append(chunk)
            if _rms(chunk) >= noise_floor:
                if last_speech is None:
                    logger.info("[record] first speech at +%.1fs", time.time() - start)
                last_speech = now
        if last_speech is not None and (now - last_speech) >= silence_seconds:
            logger.info("[record] silence reached, %d chunks", len(chunks))
            break

    total = b"".join(chunks)
    dur = len(total) / 2 / ws.server_sample_rate if ws.server_sample_rate else 0
    logger.info("[record] captured %d chunks, %.1f s%s",
                len(chunks), dur, "" if was_connected else " — disconnected")
    return total


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def run_pipeline(text: str, character: str, token: str) -> tuple[bytes, int]:
    """Run the full TTS → WebSocket → record → reply pipeline.

    Returns (pcm_bytes, sample_rate).  Raises RuntimeError on failure.
    """
    # 1 — Synthesize TTS
    logger.info("Synthesizing TTS ...")
    with tempfile.TemporaryDirectory() as td:
        tts_wav = os.path.join(td, "tts.wav")
        _synthesize_tts(text, tts_wav)
        src_rate, audio_i16 = _read_wav_mono_pcm16(tts_wav)

    audio_i16 = _resample(audio_i16, src_rate, SAMPLE_RATE)
    audio_i16 = _normalize(audio_i16)
    audio_i16 = compress_silence(audio_i16, sample_rate=SAMPLE_RATE, max_pause_ms=200)
    tts_bytes = audio_i16.tobytes()
    logger.info("TTS ready: %.1f s", audio_i16.size / SAMPLE_RATE)

    # 2 — Connect + drain greeting
    ws = SesameWebSocket(
        id_token=token,
        character=character,
        client_sample_rate=SAMPLE_RATE,
    )

    logger.info("Connecting to %s ...", character)
    if not ws.connect(blocking=True) or not ws.is_connected():
        raise RuntimeError("WebSocket connection failed.")
    logger.info("Connected (rate=%d Hz)", ws.server_sample_rate)

    _drain_until_silence(ws, label="greeting")

    if not ws.is_connected():
        raise RuntimeError("Server disconnected during greeting.")

    # 3 — Stream TTS
    logger.info("Streaming TTS ...")
    _send_silence(ws, 0.3, SAMPLE_RATE)
    _stream_pcm(ws, tts_bytes, SAMPLE_RATE)
    _send_silence(ws, 2.5, SAMPLE_RATE)
    logger.info("TTS streaming complete")

    # 4 — Record reply
    logger.info("Recording reply ...")
    reply_pcm = _record_until_silence(ws)

    ws.disconnect()
    logger.info("Disconnected")

    if not reply_pcm:
        raise RuntimeError("No reply audio received.")

    reply_rate = ws.server_sample_rate
    return reply_pcm, reply_rate
