# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""xAI TTS provider — WebSocket streaming to xAI Speak API.

Wire protocol (single-utterance benchmark path):
  connect wss://api.x.ai/v1/tts?language=en&voice=<voice>&codec=pcm&sample_rate=24000
  → send text.delta(text) → send text.done
  → recv base64 audio.delta frames until audio.done → close

Auth: Authorization: Bearer <key>
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
import wave
from pathlib import Path
from urllib.parse import urlencode

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000
_XAI_TTS_WS_BASE = "wss://api.x.ai/v1/tts"


class XAITTSProvider(TTSProvider):
    """xAI TTS provider using WebSocket streaming."""

    enabled: bool = True

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        self._model = model
        self._voice = voice

        api_key_secret = settings.xai_api_key
        if api_key_secret is None:
            raise ValueError("xai_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"xai-{self._model.lower()}"

    @property
    def model(self) -> str:
        return self._model

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech via xAI WebSocket and return a TTSResult."""
        audio_chunks: list[bytes] = []
        ttfa_ms: float | None = None

        qs = urlencode({
            "language": "en",
            "voice": self._voice,
            "codec": "pcm",
            "sample_rate": SAMPLE_RATE,
            "optimize_streaming_latency": 1,
        })
        url = f"{_XAI_TTS_WS_BASE}?{qs}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            async with ws_client.connect(url, additional_headers=headers) as ws:
                # Send the full text as a single delta, then signal done.
                await ws.send(json.dumps({"type": "text.delta", "delta": text}))

                # t0 — synthesis trigger: clock starts after text is dispatched.
                start = time.monotonic()
                await ws.send(json.dumps({"type": "text.done"}))

                async for raw in ws:
                    if isinstance(raw, bytes):
                        # Unexpected binary frame — skip.
                        continue

                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")

                    if msg_type == "audio.delta":
                        audio_data = base64.b64decode(msg.get("delta", ""))
                        if len(audio_data) > 0:
                            if ttfa_ms is None:
                                ttfa_ms = (time.monotonic() - start) * 1000
                                logger.debug(
                                    "xai_ttfa",
                                    model=self._model,
                                    ttfa_ms=ttfa_ms,
                                )
                            audio_chunks.append(audio_data)

                    elif msg_type == "audio.done":
                        break

                    elif msg_type == "error":
                        error_msg = msg.get("message", "Unknown xAI TTS error")
                        logger.warning("xai_tts_error", error=error_msg)
                        return TTSResult(
                            provider="xai",
                            model=self._model,
                            voice=self._voice,
                            ttfa_ms=ttfa_ms,
                            audio_path=None,
                            error=error_msg,
                        )

        except Exception as exc:
            logger.debug("xai_tts_error", exc_info=True)
            return TTSResult(
                provider="xai",
                model=self._model,
                voice=self._voice,
                ttfa_ms=ttfa_ms,
                audio_path=None,
                error=str(exc),
            )

        audio_path = _write_wav(audio_chunks, SAMPLE_RATE) if audio_chunks else None
        return TTSResult(
            provider="xai",
            model=self._model,
            voice=self._voice,
            ttfa_ms=ttfa_ms,
            audio_path=audio_path,
            error=None,
        )


def _write_wav(chunks: list[bytes], sample_rate: int) -> Path:
    """Concatenate PCM chunks and write a WAV file to a temp location."""
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    audio_data = b"".join(chunks)
    with wave.open(tmp_name, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    return Path(tmp_name)
