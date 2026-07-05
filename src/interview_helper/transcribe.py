"""Стриминговая транскрипция чанков через faster-whisper."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .capture import SAMPLE_RATE, AudioChunk, Source


@dataclass
class Utterance:
    source: Source
    text: str
    timestamp: float


def _load_model(model_size: str):
    """CUDA при наличии GPU и extra `gpu`, иначе CPU/int8 на всех ядрах."""
    import os

    from faster_whisper import WhisperModel

    try:
        for pkg in ("cublas", "cudnn"):
            import importlib.util

            spec = importlib.util.find_spec(f"nvidia.{pkg}")
            if spec and spec.submodule_search_locations:
                bin_dir = os.path.join(spec.submodule_search_locations[0], "bin")
                os.add_dll_directory(bin_dir)
                # ctranslate2 грузит DLL обычным поиском — нужен и PATH
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        return model, "cuda"
    except Exception:
        model = WhisperModel(
            model_size, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4
        )
        return model, "cpu"


class Transcriber:
    """Копит речь по источникам и финализирует реплику по паузе.

    Тишина до начала речи не копится вовсе; реплика закрывается после
    ~pause_chunks тихих чанков подряд (пауза) или по достижении max_seconds.
    """

    def __init__(
        self,
        model_size: str = "small",
        language: str | None = None,
        pause_chunks: int = 2,
        max_seconds: float = 15.0,
    ) -> None:
        self.model, self.device = _load_model(model_size)
        self.language = language
        self.pause_chunks = pause_chunks
        self.max_seconds = max_seconds
        self._buffers: dict[Source, list[AudioChunk]] = {"loopback": [], "mic": []}
        self._has_speech: dict[Source, bool] = {"loopback": False, "mic": False}
        self._silent_run: dict[Source, int] = {"loopback": 0, "mic": 0}
        self._last_feed: dict[Source, float] = {"loopback": 0.0, "mic": 0.0}
        self._floor: dict[Source, float] = {"loopback": 0.002, "mic": 0.002}

    def _is_silent(self, source: Source, samples: np.ndarray) -> bool:
        """Тишина относительно адаптивного шумового пола источника.

        Фиксированный порог не работает: вентилятор/кулер держат RMS микрофона
        выше любой константы, и паузы не распознаются. Пол быстро опускается до
        уровня тихих чанков и медленно ползёт вверх, речь = заметно громче пола.
        """
        rms = float(np.sqrt(np.mean(samples**2)))
        self._floor[source] = min(rms + 1e-6, self._floor[source] * 1.03)
        return rms < max(0.004, self._floor[source] * 2.5)

    def feed(self, chunk: AudioChunk) -> Utterance | None:
        s = chunk.source
        self._last_feed[s] = time.time()
        if self._is_silent(s, chunk.samples):
            if not self._has_speech[s]:
                # pre-roll: держим последний тихий чанк — речь часто начинается
                # в середине чанка, и без него глотается начало фразы
                self._buffers[s] = [chunk]
                return None
            self._buffers[s].append(chunk)
            self._silent_run[s] += 1
            if self._silent_run[s] >= self.pause_chunks:
                return self._flush(s)
            return None
        self._silent_run[s] = 0
        self._has_speech[s] = True
        self._buffers[s].append(chunk)
        buffered = sum(len(c.samples) for c in self._buffers[s]) / SAMPLE_RATE
        if buffered >= self.max_seconds:  # защита от речи без пауз
            return self._flush(s)
        return None

    def flush_stale(self, max_age: float = 1.0) -> list[Utterance]:
        """Дожимает буферы, в которые давно не поступало аудио.

        Loopback в callback-режиме при тишине не шлёт кадров вообще, поэтому
        конец реплики интервьюера часто виден только по таймауту.
        """
        now = time.time()
        result = []
        for source, buf in self._buffers.items():
            if buf and self._has_speech[source] and now - self._last_feed[source] > max_age:
                utt = self._flush(source)
                if utt:
                    result.append(utt)
        return result

    def _flush(self, source: Source) -> Utterance | None:
        buf = self._buffers[source]
        self._buffers[source] = []
        self._has_speech[source] = False
        self._silent_run[source] = 0
        if not buf:
            return None
        audio = np.concatenate([c.samples for c in buf])
        ts = buf[0].timestamp
        segments, _ = self.model.transcribe(
            audio, vad_filter=True, beam_size=1, language=self.language
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return Utterance(source, text, ts) if text else None
