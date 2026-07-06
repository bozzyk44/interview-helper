"""Бенчмарк латентности: WAV -> транскрипция -> первый токен -> полный ответ."""

from __future__ import annotations

import sys
import time
import wave

import numpy as np

from .answer import Answerer
from .capture import _resample_mono
from .transcribe import Transcriber, Utterance


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python -m interview_helper.bench <file.wav>")
    path = sys.argv[1]

    t0 = time.perf_counter()
    transcriber = Transcriber()
    t_model = time.perf_counter() - t0

    with wave.open(path, "rb") as wf:
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[wf.getsampwidth()]
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype).astype(np.float32)
        audio = _resample_mono(raw / np.iinfo(dtype).max, wf.getnchannels(), wf.getframerate())

    t0 = time.perf_counter()
    segments, _ = transcriber.model.transcribe(audio, vad_filter=True, beam_size=1)
    text = " ".join(s.text.strip() for s in segments).strip()
    t_asr = time.perf_counter() - t0

    utt = Utterance("loopback", text, time.time())
    answerer = Answerer(model="sonnet", effort="low")  # дефолт пайплайна
    answerer.add(utt)

    t0 = time.perf_counter()
    t_first = None
    parts = []
    for delta in answerer.stream_answer(utt):
        if t_first is None:
            t_first = time.perf_counter() - t0
        parts.append(delta)
    t_full = time.perf_counter() - t0

    print(f"Транскрипт: {text!r}")
    print(f"Ответ: {''.join(parts)!r}")
    print(f"\nЗагрузка модели (one-time): {t_model:.2f}s")
    print(f"Транскрипция:               {t_asr:.2f}s")
    print(f"Первый токен ответа:        {t_first or float('nan'):.2f}s")
    print(f"Полный ответ:               {t_full:.2f}s")
    print(f"Сквозная (без загрузки):    {t_asr + (t_first or 0):.2f}s")


if __name__ == "__main__":
    main()
