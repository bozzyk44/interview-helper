import time

from interview_helper.answer import Answerer, extract_delta
from interview_helper.transcribe import Utterance


def _utt(text: str, source: str = "loopback") -> Utterance:
    return Utterance(source, text, time.time())


def test_extract_delta():
    line = (
        '{"type":"stream_event","event":{"type":"content_block_delta",'
        '"delta":{"type":"text_delta","text":"привет"}}}'
    )
    assert extract_delta(line) == "привет"


def test_extract_delta_ignores_other_events_and_garbage():
    assert extract_delta('{"type":"system"}') == ""
    assert extract_delta('{"type":"assistant","message":{"content":[]}}') == ""
    assert extract_delta("not json") == ""


def test_is_question_heuristics():
    a = Answerer()
    assert a.is_question(_utt("Расскажите о вашем опыте?"))
    assert a.is_question(_utt("what is a closure"))
    assert a.is_question(_utt("Подскажите, какие бывают статусы ошибок HTTP"))
    assert a.is_question(_utt("Объясните разницу между списком и кортежем"))
    assert not a.is_question(_utt("Отлично, идём дальше."))
    # whisper коверкает первое слово — ловим fuzzy-сопоставлением
    assert a.is_question(_utt("Весните разницу между авторизацией и аутентификацией."))
    assert a.is_question(_utt("Весните про кэширование в веб-приложениях."))
    assert not a.is_question(_utt("Как это работает?", source="mic"))  # свой голос — не вопрос


def test_answer_mic_mode():
    a = Answerer(answer_mic=True)
    assert a.is_question(_utt("Какие бывают статусы ошибок HTTP?", source="mic"))
    assert not a.is_question(_utt("Сейчас говорю я.", source="mic"))


def test_resolve_effort():
    from interview_helper.answer import resolve_effort

    assert resolve_effort("haiku", "high") is None  # haiku не поддерживает effort
    assert resolve_effort("sonnet", "max") == "max"
    assert resolve_effort("opus", "high") == "medium"  # opus капнут на medium
    assert resolve_effort("opus", "low") == "low"
    assert resolve_effort("sonnet", None) is None
    assert resolve_effort("sonnet", "bogus") is None


def test_history_window_trims():
    a = Answerer()
    for _ in range(50):
        a.add(_utt("x" * 500))
    assert sum(len(u.text) for u in a.history) <= 6000
