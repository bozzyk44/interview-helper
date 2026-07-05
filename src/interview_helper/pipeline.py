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
    language: str | None = None,
    answer_mic: bool = False,
) -> None:
    chunks: queue.Queue[AudioChunk] = queue.Queue()
    try:
        _run(
            emit,
            stop,
            chunks,
            input_file,
            model_size,
            mic_device,
            loopback_device,
            language,
            answer_mic,
        )
    except Exception as e:  # noqa: BLE001 — поток фоновый, ошибку показываем пользователю
        emit({"type": "status", "text": f"Ошибка: {e!r}"})
    finally:
        emit({"type": "status", "text": "Остановлено"})


def _run(
    emit: Emit,
    stop: threading.Event,
    chunks: queue.Queue[AudioChunk],
    input_file: str | None,
    model_size: str,
    mic_device: int | None,
    loopback_device: int | None,
    language: str | None,
    answer_mic: bool,
) -> None:
    emit({"type": "status", "text": f"Загружаю whisper ({model_size}, int8)..."})
    transcriber = Transcriber(model_size=model_size, language=language)
    answerer = Answerer(answer_mic=answer_mic)

    if input_file:
        capture_stop = start_file_capture(chunks, input_file)
        emit({"type": "status", "text": f"Читаю файл {input_file}"})
    else:
        capture_stop = start_live_capture(
            chunks, mic_device=mic_device, loopback_device=loopback_device
        )
        emit({"type": "status", "text": "Слушаю встречу и микрофон"})

    def handle(utt) -> None:
        emit({"type": "utterance", "source": utt.source, "text": utt.text})
        answerer.add(utt)
        if answerer.is_question(utt):
            emit({"type": "answer_start", "question": utt.text})
            for delta in answerer.stream_answer(utt):
                if stop.is_set():
                    break
                emit({"type": "answer_delta", "text": delta})
            emit({"type": "answer_end"})

    try:
        idle = 0
        while not stop.is_set():
            try:
                chunk = chunks.get(timeout=1)
            except queue.Empty:
                # тишина: дожимаем недосказанные реплики (loopback молчит без звука)
                for utt in transcriber.flush_stale():
                    handle(utt)
                idle += 1
                if input_file and idle >= 5:
                    break  # файл закончился
                continue
            idle = 0
            utt = transcriber.feed(chunk)
            if utt is not None:
                handle(utt)
    finally:
        capture_stop.set()
