# cloud/app.py — FastAPI endpoint for Sesame AI text-to-speech.

import os
import tempfile
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from sesame_ai import SesameAI, TokenManager, run_pipeline, write_wav, wav_to_mp3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

tm: TokenManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tm
    api = SesameAI()
    tm = TokenManager(api)
    logging.info("TokenManager initialized")
    yield


app = FastAPI(title="Sesame AI TTS", version="0.1.0", lifespan=lifespan)


class SpeakRequest(BaseModel):
    text: str = Field(..., description="Text to send to the AI", min_length=1)
    character: str = Field("Maya", pattern="^(Maya|Miles)$")


@app.post("/speak")
async def speak(req: SpeakRequest):
    try:
        token = tm.get_valid_token()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Auth failed: {exc}")

    try:
        reply_pcm, reply_rate = run_pipeline(req.text, req.character, token)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "reply.wav")
        mp3_path = os.path.join(td, "reply.mp3")
        write_wav(wav_path, reply_pcm, reply_rate)
        wav_to_mp3(wav_path, mp3_path)
        with open(mp3_path, "rb") as f:
            mp3_bytes = f.read()

    dur = len(reply_pcm) / 2 / reply_rate if reply_rate else 0
    logging.info("Done — %.1f s, %d bytes", dur, len(mp3_bytes))
    return Response(content=mp3_bytes, media_type="audio/mpeg")


@app.get("/health")
async def health():
    return {"status": "ok"}
