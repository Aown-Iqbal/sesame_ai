"""
speak_and_capture.py — Send text to Sesame AI, get a spoken MP3 reply.

Usage:
  python examples/speak_and_capture.py "Hello, how are you?"
  python examples/speak_and_capture.py "Tell me a story" --character Miles
  python examples/speak_and_capture.py "Hello" --out reply.mp3 --extract-longest
"""

import tempfile
import argparse
import logging
import os

import numpy as np

from sesame_ai import SesameAI, TokenManager, run_pipeline, write_wav, wav_to_mp3
from sesame_ai.audio_utils import extract_longest_utterance


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

    # Auth
    api = SesameAI()
    tm = TokenManager(api)
    try:
        token = tm.get_valid_token()
    except Exception as exc:
        logging.error("Auth failed: %s", exc)
        return 2

    # Core pipeline
    try:
        reply_pcm, reply_rate = run_pipeline(args.text, args.character, token)
    except RuntimeError as exc:
        logging.error("Pipeline failed: %s", exc)
        return 1

    # Post-process
    if args.extract_longest:
        reply_i16 = np.frombuffer(reply_pcm, dtype=np.int16)
        before = reply_i16.size
        reply_i16 = extract_longest_utterance(reply_i16, sample_rate=reply_rate)
        if reply_i16.size < before:
            logging.info("Extracted longest: %.1fs -> %.1fs",
                         before / reply_rate, reply_i16.size / reply_rate)
        reply_pcm = reply_i16.tobytes()

    # Write output
    if args.out.lower().endswith(".mp3"):
        with tempfile.TemporaryDirectory() as td:
            tmp_wav = os.path.join(td, "reply.wav")
            write_wav(tmp_wav, reply_pcm, reply_rate)
            wav_to_mp3(tmp_wav, args.out)
    else:
        write_wav(args.out, reply_pcm, reply_rate)

    reply_dur = len(reply_pcm) / 2 / reply_rate if reply_rate else 0
    logging.info("Done — %s (%.1f s)", args.out, reply_dur)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
