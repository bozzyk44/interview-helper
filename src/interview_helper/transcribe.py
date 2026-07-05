"""Стриминговая транскрипция чанков через faster-whisper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capture import SAMPLE_RATE, AudioChunk, Source


@dataclass
class Utterance:
    source: Source
    text: str
    timestamp: float


class Transcriber:
    """Копит аудио по источникам и выдаёт реплики, когда VAD видит паузу."""

    def __init__(self, model_size: str = "small", buffer_seconds: float = 3.0) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.buffer_seconds = buffer_seconds
        self._buffers: dict[Source, list[AudioChunk]] = {"loopback": [], "mic": []}

    def feed(self, chunk: AudioChunk) -> Utterance | None:
        buf = self._buffers[chunk.source]
        buf.append(chunk)
        buffered = sum(len(c.samples) for c in buf) / SAMPLE_RATE
        # финализируем реплику, когда накопили окно и последний чанк тихий (пауза в речи)
        if buffered >= self.buffer_seconds and _is_silence(chunk.samples):
            return self._flush(chunk.source)
        if buffered >= self.buffer_seconds * 3:  # защита от бесконечной речи без пауз
            return self._flush(chunk.source)
        return None

    def _flush(self, source: Source) -> Utterance | None:
        buf = self._buffers[source]
        if not buf:
            return None
        audio = np.concatenate([c.samples for c in buf])
        ts = buf[0].timestamp
        self._buffers[source] = []
        segments, _ = self.model.transcribe(audio, vad_filter=True, beam_size=1)
        text = " ".join(s.text.strip() for s in segments).strip()
        return Utterance(source, text, ts) if text else None


def _is_silence(samples: np.ndarray, threshold: float = 0.01) -> bool:
    return float(np.sqrt(np.mean(samples**2))) < threshold
