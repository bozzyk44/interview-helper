import time

from interview_helper.answer import Answerer, extract_text
from interview_helper.transcribe import Utterance


def _utt(text: str, source: str = "loopback") -> Utterance:
    return Utterance(source, text, time.time())


def test_extract_text_assistant_event():
    line = '{"type":"assistant","message":{"content":[{"type":"text","text":"привет"}]}}'
    assert extract_text(line) == "привет"


def test_extract_text_ignores_other_events_and_garbage():
    assert extract_text('{"type":"system"}') == ""
    assert extract_text("not json") == ""


def test_is_question_heuristics():
    a = Answerer()
    assert a.is_question(_utt("Расскажите о вашем опыте?"))
    assert a.is_question(_utt("what is a closure"))
    assert a.is_question(_utt("Подскажите, какие бывают статусы ошибок HTTP"))
    assert a.is_question(_utt("Объясните разницу между списком и кортежем"))
    assert not a.is_question(_utt("Отлично, идём дальше."))
    assert not a.is_question(_utt("Как это работает?", source="mic"))  # свой голос — не вопрос


def test_answer_mic_mode():
    a = Answerer(answer_mic=True)
    assert a.is_question(_utt("Какие бывают статусы ошибок HTTP?", source="mic"))
    assert not a.is_question(_utt("Сейчас говорю я.", source="mic"))


def test_history_window_trims():
    a = Answerer()
    for _ in range(50):
        a.add(_utt("x" * 500))
    assert sum(len(u.text) for u in a.history) <= 6000
