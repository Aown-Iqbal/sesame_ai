"""
Text -> audio WAV example (best-effort).

Sesame's web app is voice-first; text message schemas can vary across deployments.
This script connects, optionally captures/discards the initial greeting, then
sends your text and records the reply audio into a WAV file.

PowerShell:
  $env:SESAME_ID_TOKEN="..."
  python examples/text_to_audio.py --character Miles --text "Hi, quick question..." --out out.wav
"""

import os
import sys
import time
import wave
import argparse
import logging

from sesame_ai import SesameWebSocket


def write_wav(path: str, pcm_bytes: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_bytes)


def drain_audio(
    ws: SesameWebSocket,
    max_seconds: float,
    stop_after_silence_seconds: float,
):
    """
    Collect audio chunks until time runs out, or until silence follows audio.

    Returns:
        (pcm_bytes, received_any_audio)
    """
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

    pcm = b"".join(chunks)
    return pcm, bool(chunks)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Generate audio from text (WAV)")
    parser.add_argument("--character", default="Miles", help="Character (Miles/Maya)")
    parser.add_argument("--timezone", default="America/Chicago", help="Timezone for connect URL usercontext")
    parser.add_argument("--client-name", default="Consumer-Web-App", help="Connect URL client_name")
    parser.add_argument("--text", required=True, help="Text to speak")
    parser.add_argument("--out", default="out.wav", help="Output WAV path")
    parser.add_argument("--skip-greeting", action="store_true", help="Discard initial greeting audio before sending text")
    parser.add_argument("--greeting-max-seconds", type=float, default=12.0, help="Max time to wait for greeting audio")
    parser.add_argument("--max-seconds", type=float, default=20.0, help="Max time to record reply after sending text")
    parser.add_argument("--silence-seconds", type=float, default=2.0, help="Stop after this long of silence after audio")
    parser.add_argument(
        "--message-type",
        default="chat",
        choices=["chat", "text", "user_text"],
        help="Text message schema to use (default: chat)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

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
    )

    if not ws.connect(blocking=True) or not ws.is_connected():
        logging.error("Failed to connect (see logs above).")
        return 1

    if args.skip_greeting:
        logging.info("Connected. Waiting for and discarding greeting audio...")
        greeting_pcm, got_greeting = drain_audio(
            ws,
            max_seconds=args.greeting_max_seconds,
            stop_after_silence_seconds=args.silence_seconds,
        )
        logging.info(
            "Greeting audio discarded: got_audio=%s bytes=%d",
            got_greeting,
            len(greeting_pcm),
        )
    else:
        logging.info("Connected. Not discarding greeting audio.")

    logging.info("Sending text message_type=%s", args.message_type)
    ok = ws.send_text(args.text, message_type=args.message_type)
    logging.info("Sent text ok=%s", ok)
    if not ok:
        logging.error("Could not send text (socket not ready).")
        ws.disconnect()
        return 1

    reply_pcm, got_reply = drain_audio(
        ws,
        max_seconds=args.max_seconds,
        stop_after_silence_seconds=args.silence_seconds,
    )
    ws.disconnect()
    if not got_reply:
        logging.error(
            "No reply audio received after sending text. This deployment may require audio/WebRTC input to trigger responses."
        )
        return 1

    write_wav(args.out, reply_pcm, ws.server_sample_rate)
    logging.info("Wrote %d bytes PCM to %s @ %d Hz", len(reply_pcm), args.out, ws.server_sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

