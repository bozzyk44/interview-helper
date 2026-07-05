"""Захват аудио: WASAPI loopback (звук встречи) + микрофон, либо файл для отладки."""

from __future__ import annotations

import queue
import threading
import time
import wave
from dataclasses import dataclass
from typing import Literal

import numpy as np

SAMPLE_RATE = 16_000  # целевой sample rate для whisper
CHUNK_SECONDS = 1.0

Source = Literal["loopback", "mic"]


@dataclass
class AudioChunk:
    source: Source
    samples: np.ndarray  # float32 mono @ 16kHz
    timestamp: float


def _resample_mono(raw: np.ndarray, channels: int, rate: int) -> np.ndarray:
    if channels > 1:
        raw = raw.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        n = int(len(raw) * SAMPLE_RATE / rate)
        raw = np.interp(np.linspace(0, len(raw) - 1, n), np.arange(len(raw)), raw)
    return raw.astype(np.float32)


def _stream_worker(
    out: queue.Queue[AudioChunk], source: Source, stop: threading.Event, loopback: bool
) -> None:
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    if loopback:
        device = pa.get_default_wasapi_loopback()
    else:
        device = pa.get_default_input_device_info()
    rate = int(device["defaultSampleRate"])
    channels = int(device["maxInputChannels"])
    frames = int(rate * CHUNK_SECONDS)
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=device["index"],
        frames_per_buffer=frames,
    )
    try:
        while not stop.is_set():
            data = np.frombuffer(stream.read(frames, exception_on_overflow=False), np.float32)
            out.put(AudioChunk(source, _resample_mono(data, channels, rate), time.time()))
    finally:
        stream.close()
        pa.terminate()


def start_live_capture(out: queue.Queue[AudioChunk]) -> threading.Event:
    """Запускает оба потока захвата; возвращает Event для остановки."""
    stop = threading.Event()
    for source, loopback in (("loopback", True), ("mic", False)):
        threading.Thread(
            target=_stream_worker, args=(out, source, stop, loopback), daemon=True
        ).start()
    return stop


def start_file_capture(
    out: queue.Queue[AudioChunk], path: str, realtime: bool = True
) -> threading.Event:
    """Отладка: читает WAV чанками, помечая источник как loopback."""
    stop = threading.Event()

    def worker() -> None:
        with wave.open(path, "rb") as wf:
            rate, channels, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
            frames = int(rate * CHUNK_SECONDS)
            while not stop.is_set():
                raw = wf.readframes(frames)
                if not raw:
                    break
                dtype = {1: np.int8, 2: np.int16, 4: np.int32}[width]
                data = np.frombuffer(raw, dtype).astype(np.float32) / np.iinfo(dtype).max
                out.put(AudioChunk("loopback", _resample_mono(data, channels, rate), time.time()))
                if realtime:
                    time.sleep(CHUNK_SECONDS)

    threading.Thread(target=worker, daemon=True).start()
    return stop
