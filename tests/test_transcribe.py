from dataclasses import dataclass

from interview_helper.transcribe import _is_hallucination, _keep_segment


@dataclass
class _Seg:
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


def test_hallucination_titres_dropped():
    assert _is_hallucination("Продолжение следует...")
    assert _is_hallucination("Спасибо за просмотр!")
    assert _is_hallucination("Субтитры сделал DimaTorzok")
    assert _is_hallucination("Thanks for watching!")


def test_hallucination_punctuation_or_single_char_dropped():
    assert _is_hallucination(".")
    assert _is_hallucination("…")
    assert _is_hallucination("а")


def test_real_speech_kept():
    assert not _is_hallucination("Расскажи, как ты организуешь ретраи в пайплайне?")
    assert not _is_hallucination("Какие типы индексов в Postgres используешь?")


def test_marker_in_long_real_sentence_not_dropped():
    # слово-маркер, но длинная живая реплика по делу — не режем
    text = (
        "У нас была задача генерировать субтитры для видео через whisper, "
        "и мы столкнулись с проблемой латентности на длинных файлах."
    )
    assert not _is_hallucination(text)


def test_keep_segment_drops_low_confidence_noise():
    # шум прошёл VAD: и «не речь», и низкий logprob одновременно
    assert not _keep_segment(_Seg(no_speech_prob=0.9, avg_logprob=-1.2))


def test_keep_segment_keeps_quiet_real_speech():
    # тихая, но уверенная речь — оставляем (нужны оба условия для отсева)
    assert _keep_segment(_Seg(no_speech_prob=0.9, avg_logprob=-0.3))
    assert _keep_segment(_Seg(no_speech_prob=0.2, avg_logprob=-1.2))
    assert _keep_segment(_Seg(no_speech_prob=0.1, avg_logprob=-0.2))
