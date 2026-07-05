"""Запись фикстуры с микрофона: python -m interview_helper.record <out.wav> [секунды]."""

from __future__ import annotations

import sys
import wave

import numpy as np

from .capture import SAMPLE_RATE, _resample_mono


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/question.wav"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    device = pa.get_default_input_device_info()
    rate = int(device["defaultSampleRate"])
    channels = int(device["maxInputChannels"])
    frames = int(rate * 0.5)
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=device["index"],
        frames_per_buffer=frames,
    )
    print(f"Пишу {seconds:.0f} c с «{device['name']}»... Говорите. (Ctrl+C — закончить раньше)")
    parts: list[np.ndarray] = []
    try:
        for i in range(int(seconds * 2)):
            data = np.frombuffer(stream.read(frames, exception_on_overflow=False), np.float32)
            parts.append(_resample_mono(data, channels, rate))
            if i % 10 == 9:
                print(f"  {((i + 1) / 2):.0f} c")
    except KeyboardInterrupt:
        print("Остановлено.")
    finally:
        stream.close()
        pa.terminate()

    audio = np.concatenate(parts)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    peak = float(np.abs(audio).max())
    print(f"Сохранено: {out} ({len(audio) / SAMPLE_RATE:.1f} c, пик {peak:.2f})")
    if peak < 0.05:
        print("ВНИМАНИЕ: сигнал очень тихий — проверьте микрофон и перезапишите.")


if __name__ == "__main__":
    main()
