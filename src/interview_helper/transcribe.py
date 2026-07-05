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
        self._last_feed: dict[Source, float] = {"loopback": 0.0, "mic": 0.0}

    def feed(self, chunk: AudioChunk) -> Utterance | None:
        import time

        buf = self._buffers[chunk.source]
        buf.append(chunk)
        self._last_feed[chunk.source] = time.time()
        buffered = sum(len(c.samples) for c in buf) / SAMPLE_RATE
        # финализируем реплику, когда накопили окно и последний чанк тихий (пауза в речи)
        if buffered >= self.buffer_seconds and _is_silence(chunk.samples):
            return self._flush(chunk.source)
        if buffered >= self.buffer_seconds * 3:  # защита от бесконечной речи без пауз
            return self._flush(chunk.source)
        return None

    def flush_stale(self, max_age: float = 1.5) -> list[Utterance]:
        """Дожимает буферы, в которые давно не поступало аудио.

        Loopback в callback-режиме при тишине не шлёт кадров вообще, поэтому
        конец реплики интервьюера виден только по таймауту.
        """
        import time

        now = time.time()
        result = []
        for source, buf in self._buffers.items():
            if buf and now - self._last_feed[source] > max_age:
                utt = self._flush(source)
                if utt:
                    result.append(utt)
        return result

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
