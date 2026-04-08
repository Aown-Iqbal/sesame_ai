"""
Record an entire Sesame session to ONE file (a .zip archive).

The archive includes:
  - events.jsonl : timestamped inbound WS messages + our outbound actions
  - server_audio.wav : all server audio chunks concatenated
  - sent_tts.wav : the exact local TTS audio we streamed as input (if used)

PowerShell:
  $env:SESAME_ID_TOKEN="..."
  python examples/record_full_exchange.py --character Miles --timezone "Asia/Karachi" --text "Hey..." --out session.zip
"""

import os
import sys
import io
import time
import json
import wave
import base64
import zipfile
import argparse
import logging
import tempfile
import math

import numpy as np

from sesame_ai import SesameWebSocket


def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)
    return buf.getvalue()


def _now_ms(start: float) -> int:
    return int((time.time() - start) * 1000)


def _read_wav_mono_pcm16(path: str):
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise ValueError("Expected 16-bit WAV")
    audio = np.frombuffer(frames, dtype=np.int16)
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return rate, audio


def _resample_linear(audio_i16: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio_i16.size == 0:
        return audio_i16
    duration = audio_i16.size / float(src_rate)
    dst_len = max(1, int(math.ceil(duration * dst_rate)))
    x_old = np.linspace(0.0, duration, num=audio_i16.size, endpoint=False)
    x_new = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    y_new = np.interp(x_new, x_old, audio_i16.astype(np.float32)).astype(np.int16)
    return y_new


def _normalize_peak(audio_i16: np.ndarray, target_peak: int = 14000) -> np.ndarray:
    if audio_i16.size == 0:
        return audio_i16
    peak = int(np.max(np.abs(audio_i16)))
    if peak <= 0:
        return audio_i16
    scale = float(target_peak) / float(peak)
    return np.clip(audio_i16.astype(np.float32) * scale, -32768, 32767).astype(np.int16)


def _synthesize_tts_to_wav(text: str, wav_path: str) -> None:
    import pyttsx3  # type: ignore
    engine = pyttsx3.init()
    engine.save_to_file(text, wav_path)
    engine.runAndWait()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Record a full exchange to one .zip")
    p.add_argument("--character", default="Miles")
    p.add_argument("--timezone", default="America/Chicago")
    p.add_argument("--client-name", default="Consumer-Web-App")
    p.add_argument("--text", required=True)
    p.add_argument("--out", default="session.zip")
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

    start = time.time()
    events = []
    server_audio_chunks = []

    def log_event(kind: str, payload: dict):
        events.append({"t_ms": _now_ms(start), "kind": kind, "payload": payload})

    ws = SesameWebSocket(
        id_token=token,
        character=args.character,
        client_name=args.client_name,
        usercontext={"timezone": args.timezone},
    )

    def on_raw(msg: dict):
        # Keep audio out of the JSONL by default; we log sizes + sample message.
        m = dict(msg)
        if m.get("type") == "audio":
            try:
                b64 = (m.get("content") or {}).get("audio_data") or ""
                log_event("ws_in_audio", {"b64_len": len(b64)})
            except Exception:
                log_event("ws_in_audio", {"b64_len": None})
        else:
            log_event("ws_in", {"type": m.get("type"), "msg": m})

    ws.set_raw_message_callback(on_raw)

    if not ws.connect(blocking=True) or not ws.is_connected():
        log_event("error", {"message": "connect_failed"})
        _write_zip(args.out, events, b"", 24000, None)
        logging.error("Failed to connect.")
        return 1

    log_event("connected", {"session_id": ws.session_id, "call_id": ws.call_id})

    def drain_server_audio(max_seconds: float):
        end = time.time() + max_seconds
        last_audio_time = None
        while time.time() < end:
            chunk = ws.get_next_audio_chunk(timeout=0.25)
            if chunk:
                server_audio_chunks.append(chunk)
                last_audio_time = time.time()
            else:
                if last_audio_time and (time.time() - last_audio_time) >= args.silence_seconds:
                    break

    if args.skip_greeting:
        log_event("phase", {"name": "discard_greeting"})
        drain_server_audio(args.greeting_max_seconds)
        server_audio_chunks.clear()

    # Create local TTS and stream as audio input (voice-first turn-taking)
    sent_tts_wav_bytes = None
    with tempfile.TemporaryDirectory() as td:
        tts_wav = os.path.join(td, "tts.wav")
        try:
            log_event("phase", {"name": "synthesize_tts"})
            _synthesize_tts_to_wav(args.text, tts_wav)
        except Exception as e:
            log_event("error", {"message": "tts_failed", "detail": str(e)})
            ws.disconnect()
            _write_zip(args.out, events, b"", ws.server_sample_rate, None)
            logging.error("Local TTS failed. Install: python -m pip install pyttsx3")
            return 1

        src_rate, audio_i16 = _read_wav_mono_pcm16(tts_wav)
        audio_16k = _resample_linear(audio_i16, src_rate, 16000)
        audio_16k = _normalize_peak(audio_16k, 14000)
        sent_tts_wav_bytes = _wav_bytes(audio_16k.tobytes(), 16000)
        log_event("tts_prepared", {"src_rate": src_rate, "samples_16k": int(audio_16k.size)})

        # leading silence
        log_event("ws_out_audio", {"note": "leading_silence", "seconds": 0.4})
        silence = np.zeros(int(0.4 * 16000), dtype=np.int16).tobytes()
        _stream_bytes(ws, silence, 16000)

        log_event("ws_out_audio", {"note": "tts_audio", "bytes": len(audio_16k.tobytes())})
        _stream_bytes(ws, audio_16k.tobytes(), 16000)

        # trailing silence
        log_event("ws_out_audio", {"note": "trailing_silence", "seconds": 2.5})
        silence2 = np.zeros(int(2.5 * 16000), dtype=np.int16).tobytes()
        _stream_bytes(ws, silence2, 16000)

    log_event("phase", {"name": "record_reply"})
    drain_server_audio(args.reply_max_seconds)
    ws.disconnect()

    server_pcm = b"".join(server_audio_chunks)
    server_wav = _wav_bytes(server_pcm, ws.server_sample_rate)

    _write_zip(args.out, events, server_wav, ws.server_sample_rate, sent_tts_wav_bytes)
    logging.info("Wrote %s (events.jsonl + server_audio.wav + sent_tts.wav)", args.out)
    return 0


def _stream_bytes(ws: SesameWebSocket, pcm_bytes: bytes, sample_rate: int, chunk_ms: int = 20):
    samples_per_chunk = int(sample_rate * (chunk_ms / 1000.0))
    step = max(1, samples_per_chunk) * 2
    for i in range(0, len(pcm_bytes), step):
        ws.send_audio_data(pcm_bytes[i : i + step])
        time.sleep(chunk_ms / 1000.0)


def _write_zip(out_path: str, events: list, server_wav: bytes, server_rate: int, sent_tts_wav: bytes | None):
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # JSONL
        jsonl = "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n"
        z.writestr("events.jsonl", jsonl.encode("utf-8"))
        if server_wav:
            z.writestr("server_audio.wav", server_wav)
        if sent_tts_wav:
            z.writestr("sent_tts.wav", sent_tts_wav)


if __name__ == "__main__":
    raise SystemExit(main())

