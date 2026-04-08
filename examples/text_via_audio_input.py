"""
Text -> (local TTS) -> audio-in -> Sesame voice reply (WAV)

Why:
Sesame's consumer WebSocket is voice-first. In many deployments, arbitrary text
JSON messages are ignored for turn-taking, and the agent waits for AUDIO input.

This script uses a *local* Windows TTS engine (pyttsx3 / SAPI5) to speak your
text into a WAV, resamples to 16kHz mono 16-bit PCM, streams it as audio input,
then records Sesame's spoken reply audio to a WAV file.

Install:
  python -m pip install pyttsx3

PowerShell:
  $env:SESAME_ID_TOKEN="..."
  python examples/text_via_audio_input.py --character Miles --timezone "Asia/Karachi" --text "Hey, quick question..." --out reply.wav --skip-greeting
"""

import os
import sys
import time
import wave
import math
import tempfile
import argparse
import logging

import numpy as np
import shutil
import subprocess

from sesame_ai import SesameWebSocket


def write_wav(path: str, pcm_bytes: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_bytes)


def wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "128k") -> None:
    """
    Convert WAV -> MP3 using ffmpeg (must be installed and on PATH).
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                wav_path,
                "-vn",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                bitrate,
                mp3_path,
            ],
            check=True,
        )
        return

    # Lightweight fallback: LAME encoder
    lame = shutil.which("lame")
    if lame:
        # Convert "128k" -> "128" for lame -b
        br = bitrate.lower().replace("k", "")
        subprocess.run(
            [
                lame,
                "--silent",
                "-b",
                br,
                wav_path,
                mp3_path,
            ],
            check=True,
        )
        return

    raise RuntimeError(
        "No MP3 encoder found. Install ONE of:\n"
        "- ffmpeg (bigger): `winget install -e --id Gyan.FFmpeg.Essentials`\n"
        "- lame (smaller): `winget install -e --id LAME.LAME`\n"
        "Or choose a .wav output."
    )

def audio_stats(audio_i16: np.ndarray) -> dict:
    if audio_i16.size == 0:
        return {"samples": 0, "peak": 0, "rms": 0.0}
    peak = int(np.max(np.abs(audio_i16)))
    rms = float(np.sqrt(np.mean(audio_i16.astype(np.float32) ** 2)))
    return {"samples": int(audio_i16.size), "peak": peak, "rms": rms}


def drain_audio(ws: SesameWebSocket, max_seconds: float, stop_after_silence_seconds: float) -> bytes:
    start = time.time()
    last_audio_time = None
    chunks = []
    while True:
        now = time.time()
        if now - start > max_seconds:
            break
        chunk = ws.get_next_audio_chunk(timeout=0.25)
        if chunk:
            chunks.append(chunk)
            last_audio_time = now
        else:
            if last_audio_time is not None and (now - last_audio_time) >= stop_after_silence_seconds:
                break
    return b"".join(chunks)


def read_wav_mono_pcm16(path: str):
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit WAV (sampwidth=2), got {sampwidth}")

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif channels != 1:
        raise ValueError(f"Unsupported channels: {channels}")

    return rate, audio


def resample_linear(audio_i16: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio_i16
    if audio_i16.size == 0:
        return audio_i16
    duration = audio_i16.size / float(src_rate)
    dst_len = max(1, int(math.ceil(duration * dst_rate)))
    x_old = np.linspace(0.0, duration, num=audio_i16.size, endpoint=False)
    x_new = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    y_old = audio_i16.astype(np.float32)
    y_new = np.interp(x_new, x_old, y_old).astype(np.int16)
    return y_new


def normalize_peak(audio_i16: np.ndarray, target_peak: int = 12000) -> np.ndarray:
    """
    Simple peak normalization to make sure the input isn't too quiet.
    """
    if audio_i16.size == 0:
        return audio_i16
    peak = int(np.max(np.abs(audio_i16)))
    if peak <= 0:
        return audio_i16
    scale = float(target_peak) / float(peak)
    if scale <= 0:
        return audio_i16
    scaled = np.clip(audio_i16.astype(np.float32) * scale, -32768, 32767).astype(np.int16)
    return scaled


def synthesize_tts_to_wav(text: str, wav_path: str) -> None:
    try:
        import pyttsx3  # type: ignore
    except Exception as e:
        raise RuntimeError("pyttsx3 is required. Install with: python -m pip install pyttsx3") from e

    engine = pyttsx3.init()
    engine.save_to_file(text, wav_path)
    engine.runAndWait()


def stream_pcm_to_sesame(ws: SesameWebSocket, pcm16: np.ndarray, sample_rate: int, chunk_ms: int = 20) -> None:
    # Sesame voice_chat uses 16kHz 16-bit mono PCM chunks. We'll send ~20ms chunks.
    samples_per_chunk = int(sample_rate * (chunk_ms / 1000.0))
    if samples_per_chunk <= 0:
        samples_per_chunk = 320
    audio_bytes = pcm16.tobytes()
    step = samples_per_chunk * 2  # int16 bytes
    for i in range(0, len(audio_bytes), step):
        ok = ws.send_audio_data(audio_bytes[i : i + step])
        if not ok:
            raise ConnectionError("WebSocket is not connected while streaming audio")
        time.sleep(chunk_ms / 1000.0)


def send_silence(ws: SesameWebSocket, seconds: float, sample_rate: int = 16000, chunk_ms: int = 20) -> None:
    """
    Send trailing silence so the server can detect end-of-utterance.
    """
    samples = int(seconds * sample_rate)
    silence = np.zeros(samples, dtype=np.int16)
    stream_pcm_to_sesame(ws, silence, sample_rate=sample_rate, chunk_ms=chunk_ms)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Send text via local TTS audio input, record Sesame reply (WAV)")
    p.add_argument("--character", default="Miles")
    p.add_argument("--timezone", default="America/Chicago")
    p.add_argument("--client-name", default="Consumer-Web-App")
    p.add_argument("--text", required=True)
    p.add_argument("--out", default="reply.wav", help="Output file (.wav or .mp3)")
    p.add_argument("--mp3-bitrate", default="128k", help="MP3 bitrate if --out ends with .mp3 (default: 128k)")
    p.add_argument("--debug-tts-wav", default=None, help="Optional path to write the TTS audio we send (for debugging)")
    p.add_argument("--input-rate", type=int, default=44100, help="Sample rate to declare + stream as (try 44100 to match web app)")
    p.add_argument("--skip-greeting", action="store_true")
    p.add_argument("--greeting-max-seconds", type=float, default=12.0)
    p.add_argument("--reply-max-seconds", type=float, default=20.0)
    p.add_argument("--silence-seconds", type=float, default=2.0)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if args.debug:
        logging.getLogger("sesame").setLevel(logging.DEBUG)
        logging.getLogger("websocket").setLevel(logging.INFO)

    token = os.environ.get("SESAME_ID_TOKEN")
    if not token:
        print("Missing SESAME_ID_TOKEN in environment.", file=sys.stderr)
        return 2

    ws = SesameWebSocket(
        id_token=token,
        character=args.character,
        client_name=args.client_name,
        usercontext={"timezone": args.timezone},
        client_sample_rate=args.input_rate,
    )
    if not ws.connect(blocking=True) or not ws.is_connected():
        logging.error("Failed to connect (see logs above).")
        return 1

    if args.skip_greeting:
        logging.info("Discarding greeting audio...")
        _ = drain_audio(ws, max_seconds=args.greeting_max_seconds, stop_after_silence_seconds=args.silence_seconds)

    with tempfile.TemporaryDirectory() as td:
        tts_wav = os.path.join(td, "tts.wav")
        logging.info("Synthesizing local TTS...")
        synthesize_tts_to_wav(args.text, tts_wav)

        src_rate, audio_i16 = read_wav_mono_pcm16(tts_wav)
        audio_out = resample_linear(audio_i16, src_rate=src_rate, dst_rate=args.input_rate)
        audio_out = normalize_peak(audio_out, target_peak=14000)

    st = audio_stats(audio_out)
    logging.info(
        "Prepared TTS audio: samples=%d peak=%d rms=%.1f (~%.2fs) rate=%d",
        st["samples"],
        st["peak"],
        st["rms"],
        st["samples"] / float(args.input_rate),
        args.input_rate,
    )
    if args.debug_tts_wav:
        write_wav(args.debug_tts_wav, audio_out.tobytes(), args.input_rate)
        logging.info("Wrote debug TTS WAV to %s", args.debug_tts_wav)

    logging.info("Streaming audio input to Sesame (%d samples @ %d Hz)...", audio_out.size, args.input_rate)
    try:
        # A short leading silence can help the server's VAD start cleanly.
        logging.info("Sending leading silence...")
        send_silence(ws, seconds=0.4, sample_rate=args.input_rate, chunk_ms=20)
        stream_pcm_to_sesame(ws, audio_out, sample_rate=args.input_rate, chunk_ms=20)
        # Trailing silence helps the server finalize the user turn.
        logging.info("Sending trailing silence...")
        send_silence(ws, seconds=2.5, sample_rate=args.input_rate, chunk_ms=20)
    except ConnectionError as e:
        logging.error("Audio streaming aborted: %s", e)
        # If server closed connection mid-stream, no point continuing.
        try:
            ws.disconnect()
        except Exception:
            pass
        return 1

    logging.info("Recording Sesame reply...")
    reply_pcm = drain_audio(ws, max_seconds=args.reply_max_seconds, stop_after_silence_seconds=args.silence_seconds)
    ws.disconnect()

    if not reply_pcm:
        logging.error("No reply audio received.")
        return 1

    if args.out.lower().endswith(".mp3"):
        # Write WAV to a temp file, then transcode to MP3.
        with tempfile.TemporaryDirectory() as td:
            tmp_wav = os.path.join(td, "reply.wav")
            write_wav(tmp_wav, reply_pcm, ws.server_sample_rate)
            wav_to_mp3(tmp_wav, args.out, bitrate=args.mp3_bitrate)
        logging.info("Wrote reply to %s (mp3)", args.out)
    else:
        write_wav(args.out, reply_pcm, ws.server_sample_rate)
        logging.info("Wrote reply to %s @ %d Hz", args.out, ws.server_sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

