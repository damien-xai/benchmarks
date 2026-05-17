# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""xAI real-time STT provider.

Wire protocol: WebSocket, wss://api.x.ai/v1/stt
Auth: Authorization: Bearer <key>
Audio: raw binary PCM chunks (no base64 wrapping)
Close: {"type": "audio.done"}

Server events:
  transcript.created  — ready signal (wait before sending audio)
  transcript.partial  — interim / chunk-final / utterance-final
  transcript.done     — full final transcript, connection closes after
"""

from __future__ import annotations

import asyncio
import json
import time  # monotonic clock — wall-clock can step on NTP sync
from typing import Any

import structlog
import websockets.asyncio.client as ws_client
from pydantic import SecretStr

from coval_bench.providers.base import STTProvider, TranscriptionResult

logger = structlog.get_logger(__name__)


class XAISTTProvider(STTProvider):
    """xAI streaming STT provider."""

    def __init__(self, api_key: SecretStr, model: str = "Grok") -> None:
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return "xai"

    @property
    def model(self) -> str:
        return self._model

    def _build_websocket_url(self, sample_rate: int) -> str:
        return (
            "wss://api.x.ai/v1/stt"
            f"?sample_rate={sample_rate}"
            "&encoding=pcm"
            "&interim_results=true"
            "&language=en"
            "&endpointing=0"
        )

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
        audio_duration: float | None = None,
    ) -> TranscriptionResult:
        result = TranscriptionResult(provider=self.name, vad_events_count=0)
        total_start = time.monotonic()

        try:
            url = self._build_websocket_url(sample_rate)
            headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

            async with ws_client.connect(url, additional_headers=headers) as ws:
                # Wait for transcript.created (server ready signal)
                try:
                    raw_ready = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    ready_msg: dict[str, Any] = json.loads(raw_ready)
                    if ready_msg.get("type") != "transcript.created":
                        logger.warning(
                            "unexpected first message from xai stt",
                            msg=ready_msg,
                        )
                except TimeoutError:
                    logger.warning("timeout waiting for xai transcript.created")
                except Exception as exc:
                    logger.exception("error reading xai transcript.created", error=str(exc))
                    raise

                send_task = asyncio.create_task(
                    self._send_audio(
                        ws,
                        audio_data,
                        channels,
                        sample_width,
                        sample_rate,
                        result,
                        realtime_resolution,
                    )
                )
                recv_task = asyncio.create_task(self._receive(ws, result))
                await asyncio.gather(send_task, recv_task, return_exceptions=True)

        except Exception as exc:
            logger.exception("xai measure_ttft failed", error=str(exc))
            result.error = str(exc)

        result.total_time = time.monotonic() - total_start
        return result

    async def _send_audio(
        self,
        ws: Any,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        result: TranscriptionResult,
        realtime_resolution: float,
    ) -> None:
        byte_rate = sample_width * sample_rate * channels
        data = audio_data
        first_chunk = True
        try:
            while data:
                chunk_size = int(byte_rate * realtime_resolution)
                chunk, data = data[:chunk_size], data[chunk_size:]
                if first_chunk:
                    result.audio_start_time = time.monotonic()
                    first_chunk = False
                await ws.send(chunk)
                await asyncio.sleep(realtime_resolution)
            # Signal end of audio
            await ws.send(json.dumps({"type": "audio.done"}))
        except Exception as exc:
            logger.exception("xai send error", error=str(exc))
            raise

    async def _receive(self, ws: Any, result: TranscriptionResult) -> None:
        final_segments: list[str] = []
        last_final_time: float | None = None

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg: dict[str, Any] = json.loads(raw)
                now = time.monotonic()
                msg_type: str = msg.get("type", "")

                if msg_type == "error":
                    error_text = msg.get("message", "Unknown xAI STT error")
                    result.error = error_text
                    logger.error("xai_stt_error", error=error_text)
                    break

                if msg_type == "transcript.partial":
                    transcript = msg.get("text", "").strip()
                    is_final = msg.get("is_final", False)
                    speech_final = msg.get("speech_final", False)

                    if not transcript:
                        continue

                    # TTFT — first text from the server
                    if result.ttft_seconds is None and result.audio_start_time is not None:
                        result.ttft_seconds = now - result.audio_start_time
                        result.first_token_content = (
                            transcript[:30] + "..." if len(transcript) > 30 else transcript
                        )

                    result.partial_transcripts.append(transcript)

                    if speech_final:
                        final_segments.append(transcript)
                        last_final_time = now

                elif msg_type == "transcript.done":
                    # Final transcript for the session
                    transcript = msg.get("text", "").strip()
                    if transcript:
                        result.complete_transcript = transcript
                        if result.audio_start_time is not None:
                            last_final_time = now
                    break

                elif msg_type == "transcript.created":
                    # Late ready signal — ignore
                    continue

        except Exception as exc:
            logger.exception("xai receive error", error=str(exc))

        if last_final_time is not None and result.audio_start_time is not None:
            result.audio_to_final_seconds = last_final_time - result.audio_start_time

        # Build complete transcript from transcript.done or fall back to finals/partials
        if result.complete_transcript is None:
            if final_segments:
                result.complete_transcript = " ".join(final_segments).strip() or None
            elif result.partial_transcripts:
                result.complete_transcript = (
                    max(result.partial_transcripts, key=len).strip() or None
                )

        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
