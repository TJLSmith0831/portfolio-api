import pytest

from app.api import player_fit_summarizer as summarizer
from app.api.player_fit_summarizer import _chat_with_retries
from app.models.player_fit_models import PlayerFitRequest, RelevantInfoResponse


class DummyMessage:
    def __init__(self, content: str):
        self.content = content


class DummyChoice:
    def __init__(self, content: str):
        self.message = DummyMessage(content)


class DummyResponse:
    def __init__(self, text: str):
        self.choices = [DummyChoice(text)]


class DummyClient:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls = []

    def chat(self, *, model, messages):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
            }
        )
        if not self._responses:
            raise AssertionError("No more responses configured for DummyClient.")
        return DummyResponse(self._responses.pop(0))


def test_extract_json_raises_when_no_json_found():
    with pytest.raises(ValueError):
        summarizer._extract_json("Completely invalid content without braces")


def test_generate_with_retries_returns_on_first_success():
    client = DummyClient(['{"status": "ok"}'])

    output = _chat_with_retries(
        client=client,
        model="some-model",
        system="system-prompt",
        messages=[{"role": "user", "content": "initial-prompt"}],
    )

    assert output == {"status": "ok"}
    assert len(client.calls) == 1

    sent_messages = client.calls[0]["messages"]

    assert sent_messages[0]["role"] == "system"
    assert sent_messages[0]["content"] == "system-prompt"

    assert sent_messages[1]["role"] == "user"
    assert sent_messages[1]["content"] == "initial-prompt"


def test_generate_with_retries_retries_and_repairs_prompt():
    client = DummyClient([
        "not json",
        '{"fixed": true}',
    ])

    output = _chat_with_retries(
        client=client,
        model="some-model",
        system="system-prompt",
        messages=[{"role": "user", "content": "initial-prompt"}],
    )

    assert output == {"fixed": True}
    assert len(client.calls) == 2

    first_messages = client.calls[0]["messages"]

    assert first_messages[0]["content"] == "system-prompt"
    assert first_messages[1]["content"] == "initial-prompt"

    repair_messages = client.calls[1]["messages"]

    assert repair_messages[0]["content"] == "system-prompt"
    repair_prompt = repair_messages[1]["content"]

    assert repair_prompt.startswith("The previous response was not valid JSON.")
    assert "he previous response was not valid JSON" in repair_prompt


def test_generate_with_retries_raises_after_exhausting_attempts():
    client = DummyClient([
        "still not json",
        "also not json",
    ])

    with pytest.raises(RuntimeError) as exc_info:
        _chat_with_retries(
            client=client,
            model="some-model",
            system="system-prompt",
            messages=[{"role": "user", "content": "initial-prompt"}],
            max_retries=1,
        )

    assert "Failed after 2 attempts" in str(exc_info.value)
    assert len(client.calls) == 2


def test_get_player_fit_prompt_includes_request_details():
    profile = RelevantInfoResponse(identity={"name": "Sample Player"})

    request = PlayerFitRequest(
        player_name="Jane Doe",
        team_name="Example College",
        player_profile=profile,
    )

    summarizer_object = summarizer.PlayerFitSummarizer()
    prompt = summarizer_object._build_player_fit_prompt(request)

    assert "Jane Doe" in prompt
    assert "Example College" in prompt
    assert request.player_profile.model_dump_json(indent=2) in prompt
    assert 'Player: Jane Doe' in prompt
    assert 'Team: Example College' in prompt
