"""
speak_and_capture.py — Send text to Sesame AI, get a spoken MP3 reply.

Usage:
  python examples/speak_and_capture.py "Hello, how are you?"
  python examples/speak_and_capture.py "Tell me a story" --character Miles
  python examples/speak_and_capture.py "Hello" --out reply.mp3 --extract-longest
"""

import os
import sys
import time
import wave
import math
import tempfile
import argparse
import logging
import subprocess
import shutil

import numpy as np

from sesame_ai import SesameAI, SesameWebSocket, TokenManager
from sesame_ai.audio_utils import compress_silence, extract_longest_utterance

# ---------------------------------------------------------------------------
# helpers
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


def _write_wav(path: str, pcm: bytes, rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(rate))
        wf.writeframes(pcm)


def _wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "96k") -> None:
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


def _synthesize_tts(text: str, wav_path: str) -> None:
    try:
        import pyttsx3
    except Exception:
        raise RuntimeError("pyttsx3 is required.  pip install pyttsx3")
    engine = pyttsx3.init()
    engine.save_to_file(text, wav_path)
    engine.runAndWait()


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
# core: drain & record
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

    logging.info("[%s] waiting for greeting to end ...", label)

    while True:
        if time.time() - start > max_wait:
            logging.info("[%s] max wait reached, %d chunks discarded", label, discarded)
            break
        if not ws.is_connected():
            logging.warning("[%s] disconnected during drain", label)
            break
        if (time.time() - last_ping) >= 10.0:
            ws.ping()
            last_ping = time.time()

        chunk = ws.get_next_audio_chunk(timeout=0.5)
        if chunk:
            discarded += 1
            if last_audio is None:
                logging.info("[%s] first audio at +%.1fs", label, time.time() - start)
            last_audio = time.time()
        elif last_audio is not None:
            if (time.time() - last_audio) >= silence_seconds:
                logging.info("[%s] silence reached, %d chunks discarded", label, discarded)
                break

    return discarded


def _record_until_silence(ws: SesameWebSocket, *,
                          max_duration: float = 180.0,
                          silence_seconds: float = 3.0) -> bytes:
    """Record audio chunks until silence or max duration."""
    start = time.time()
    last_audio: float | None = None
    last_ping = start
    chunks: list[bytes] = []
    was_connected = True

    logging.info("[record] listening ...")

    while True:
        now = time.time()
        if now - start > max_duration:
            logging.info("[record] max duration reached")
            break
        if not ws.is_connected():
            logging.warning("[record] server disconnected — %d chunks captured", len(chunks))
            was_connected = False
            break
        if (now - last_ping) >= 10.0:
            ws.ping()
            last_ping = now

        chunk = ws.get_next_audio_chunk(timeout=0.5)
        if chunk:
            chunks.append(chunk)
            if last_audio is None:
                logging.info("[record] first audio at +%.1fs", time.time() - start)
            last_audio = now
        elif last_audio is not None:
            if (now - last_audio) >= silence_seconds:
                logging.info("[record] silence reached, %d chunks", len(chunks))
                break

    total = b"".join(chunks)
    dur = len(total) / 2 / ws.server_sample_rate if ws.server_sample_rate else 0
    logging.info("[record] captured %d chunks, %.1f s%s",
                 len(chunks), dur, "" if was_connected else " — disconnected")
    return total


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Send text to Sesame AI, get MP3 reply")
    p.add_argument("text", help="Text to send (spoken via local TTS)")
    p.add_argument("--character", default="Maya", choices=["Maya", "Miles"],
                   help="AI character (default: Maya)")
    p.add_argument("--out", default="reply.mp3", help="Output file (default: reply.mp3)")
    p.add_argument("--extract-longest", action="store_true",
                   help="Post-process: keep only the longest utterance")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = p.parse_args()

    if args.debug:
        logging.getLogger("sesame").setLevel(logging.DEBUG)

    SAMPLE_RATE = 16000

    # ---- Pre-synthesize TTS ----
    logging.info("Synthesizing TTS ...")
    with tempfile.TemporaryDirectory() as td:
        tts_wav = os.path.join(td, "tts.wav")
        _synthesize_tts(args.text, tts_wav)
        src_rate, audio_i16 = _read_wav_mono_pcm16(tts_wav)

    audio_i16 = _resample(audio_i16, src_rate, SAMPLE_RATE)
    audio_i16 = _normalize(audio_i16)
    audio_i16 = compress_silence(audio_i16, sample_rate=SAMPLE_RATE, max_pause_ms=200)
    tts_bytes = audio_i16.tobytes()
    logging.info("TTS ready: %.1f s", audio_i16.size / SAMPLE_RATE)

    # ---- Auth ----
    api = SesameAI()
    tm = TokenManager(api)
    try:
        token = tm.get_valid_token()
    except Exception as exc:
        logging.error("Auth failed: %s", exc)
        return 2

    # ---- Connect + drain greeting ----
    ws = SesameWebSocket(
        id_token=token,
        character=args.character,
        client_sample_rate=SAMPLE_RATE,
    )

    logging.info("Connecting to %s ...", args.character)
    if not ws.connect(blocking=True) or not ws.is_connected():
        logging.error("Connection failed.")
        return 1
    logging.info("Connected (rate=%d Hz)", ws.server_sample_rate)

    _drain_until_silence(ws, label="greeting")

    if not ws.is_connected():
        logging.error("Server disconnected during greeting. Try again.")
        return 1

    # ---- Stream TTS ----
    logging.info("Streaming TTS ...")
    _send_silence(ws, 0.3, SAMPLE_RATE)
    _stream_pcm(ws, tts_bytes, SAMPLE_RATE)
    _send_silence(ws, 2.5, SAMPLE_RATE)
    logging.info("TTS streaming complete")

    # ---- Record reply ----
    logging.info("Recording reply ...")
    reply_pcm = _record_until_silence(ws)

    ws.disconnect()
    logging.info("Disconnected")

    if not reply_pcm:
        logging.error("No reply audio received.")
        return 1

    reply_rate = ws.server_sample_rate

    # ---- Post-process ----
    if args.extract_longest:
        reply_i16 = np.frombuffer(reply_pcm, dtype=np.int16)
        before = reply_i16.size
        reply_i16 = extract_longest_utterance(reply_i16, sample_rate=reply_rate)
        if reply_i16.size < before:
            logging.info("Extracted longest: %.1fs -> %.1fs",
                         before / reply_rate, reply_i16.size / reply_rate)
        reply_pcm = reply_i16.tobytes()

    # ---- Write output ----
    if args.out.lower().endswith(".mp3"):
        with tempfile.TemporaryDirectory() as td:
            tmp_wav = os.path.join(td, "reply.wav")
            _write_wav(tmp_wav, reply_pcm, reply_rate)
            _wav_to_mp3(tmp_wav, args.out)
    else:
        _write_wav(args.out, reply_pcm, reply_rate)

    reply_dur = len(reply_pcm) / 2 / reply_rate if reply_rate else 0
    logging.info("Done — %s (%.1f s)", args.out, reply_dur)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
