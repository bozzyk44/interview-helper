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


def list_devices() -> dict[str, list[dict]]:
    """Доступные устройства: микрофоны и WASAPI loopback-выходы."""
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    try:
        mics, loopbacks = [], []
        default_mic_info = pa.get_default_input_device_info()
        default_mic = default_mic_info["index"]
        # дефолтный вход обычно числится за MME с обрезанным именем — сопоставляем по имени
        default_mic_name = str(default_mic_info["name"])
        default_lb = pa.get_default_wasapi_loopback()["index"]
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            # только WASAPI: другие host API (MME, DirectSound) дублируют те же устройства
            if d["maxInputChannels"] < 1 or d["hostApi"] != wasapi:
                continue
            entry = {"index": i, "name": d["name"]}
            if d.get("isLoopbackDevice"):
                entry["default"] = i == default_lb
                loopbacks.append(entry)
            else:
                entry["default"] = i == default_mic or str(d["name"]).startswith(
                    default_mic_name[:31]  # MME обрезает имена до 31 символа
                )
                mics.append(entry)
        # дефолтный вход может числиться за другим host API — тогда помечаем первый WASAPI-мик
        if mics and not any(m["default"] for m in mics):
            mics[0]["default"] = True
        return {"mic": mics, "loopback": loopbacks}
    finally:
        pa.terminate()


def _live_worker(
    out: queue.Queue[AudioChunk],
    stop: threading.Event,
    mic_device: int | None,
    loopback_device: int | None,
) -> None:
    """Callback-захват обоих источников на одном PyAudio.

    Блокирующий stream.read() здесь нельзя: WASAPI loopback при тишине не отдаёт
    кадров, чтение виснет, а параллельные блокирующие потоки роняют процесс.
    """
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    acc: dict[Source, list[np.ndarray]] = {"loopback": [], "mic": []}
    lock = threading.Lock()
    streams = []
    try:
        for source, loopback, override in (
            ("loopback", True, loopback_device),
            ("mic", False, mic_device),
        ):
            if override is not None:
                device = pa.get_device_info_by_index(override)
            elif loopback:
                device = pa.get_default_wasapi_loopback()
            else:
                device = pa.get_default_input_device_info()
            rate = int(device["defaultSampleRate"])
            channels = int(device["maxInputChannels"])

            def cb(in_data, frame_count, time_info, status, s=source, ch=channels, r=rate):
                data = np.frombuffer(in_data, np.float32)
                with lock:
                    acc[s].append(_resample_mono(data, ch, r))
                return (None, pyaudio.paContinue)

            streams.append(
                pa.open(
                    format=pyaudio.paFloat32,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=device["index"],
                    frames_per_buffer=2048,
                    stream_callback=cb,
                )
            )
        target = int(SAMPLE_RATE * CHUNK_SECONDS)
        while not stop.is_set():
            time.sleep(0.2)
            for source in ("loopback", "mic"):
                with lock:
                    if sum(len(a) for a in acc[source]) < target:
                        continue
                    samples = np.concatenate(acc[source])
                    acc[source] = []
                out.put(AudioChunk(source, samples, time.time()))
    finally:
        for s in streams:
            s.stop_stream()
            s.close()
        pa.terminate()


def start_live_capture(
    out: queue.Queue[AudioChunk],
    mic_device: int | None = None,
    loopback_device: int | None = None,
) -> threading.Event:
    """Запускает захват; возвращает Event для остановки."""
    stop = threading.Event()
    threading.Thread(
        target=_live_worker, args=(out, stop, mic_device, loopback_device), daemon=True
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
