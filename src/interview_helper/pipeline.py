"""Общий цикл пайплайна: захват -> транскрипция -> ответы, с колбэком событий.

События emit(): {"type": "status"|"utterance"|"answer_start"|"answer_delta"|"answer_end", ...}
Используется и терминальным main.py, и веб-обёрткой web.py.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

from .answer import Answerer
from .capture import AudioChunk, start_file_capture, start_live_capture
from .transcribe import Transcriber

Emit = Callable[[dict], None]


def run_pipeline(
    emit: Emit,
    stop: threading.Event,
    input_file: str | None = None,
    model_size: str = "small",
    mic_device: int | None = None,
    loopback_device: int | None = None,
) -> None:
    chunks: queue.Queue[AudioChunk] = queue.Queue()
    emit({"type": "status", "text": f"Загружаю whisper ({model_size}, int8)..."})
    transcriber = Transcriber(model_size=model_size)
    answerer = Answerer()

    if input_file:
        capture_stop = start_file_capture(chunks, input_file)
        emit({"type": "status", "text": f"Читаю файл {input_file}"})
    else:
        capture_stop = start_live_capture(
            chunks, mic_device=mic_device, loopback_device=loopback_device
        )
        emit({"type": "status", "text": "Слушаю встречу и микрофон"})

    try:
        while not stop.is_set():
            try:
                chunk = chunks.get(timeout=5)
            except queue.Empty:
                if input_file:
                    break  # файл закончился
                continue
            utt = transcriber.feed(chunk)
            if utt is None:
                continue
            emit({"type": "utterance", "source": utt.source, "text": utt.text})
            answerer.add(utt)
            if answerer.is_question(utt):
                emit({"type": "answer_start", "question": utt.text})
                for delta in answerer.stream_answer(utt):
                    if stop.is_set():
                        break
                    emit({"type": "answer_delta", "text": delta})
                emit({"type": "answer_end"})
    finally:
        capture_stop.set()
        emit({"type": "status", "text": "Остановлено"})
