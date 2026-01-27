import pytest

from app.api import player_fit_summarizer as summarizer
from app.models.player_models import PlayerFitRequest, RelevantInfoResponse


class DummyResponse:
    def __init__(self, text: str) -> None:
        self.response = text


class DummyClient:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls = []

    def generate(self, *, model, system, prompt):
        self.calls.append({"model": model, "system": system, "prompt": prompt})
        if not self._responses:
            raise AssertionError("No more responses configured for DummyClient.")
        return DummyResponse(self._responses.pop(0))


def test_extract_json_raises_when_no_json_found():
    with pytest.raises(ValueError):
        summarizer._extract_json("Completely invalid content without braces")


def test_generate_with_retries_returns_on_first_success():
    client = DummyClient(['{"status": "ok"}'])
    output = summarizer._generate_with_retries(
        client=client,
        model="some-model",
        system="system-prompt",
        prompt="initial-prompt",
    )
    assert output == {"status": "ok"}
    assert len(client.calls) == 1
    assert client.calls[0]["prompt"] == "initial-prompt"


def test_generate_with_retries_retries_and_repairs_prompt():
    client = DummyClient([
        "not json",
        '{"fixed": true}',
    ])
    output = summarizer._generate_with_retries(
        client=client,
        model="some-model",
        system="system-prompt",
        prompt="initial-prompt",
    )
    assert output == {"fixed": True}
    assert len(client.calls) == 2
    assert client.calls[0]["prompt"] == "initial-prompt"
    repair_prompt = client.calls[1]["prompt"]
    assert repair_prompt.startswith("The previous response was not valid JSON.")
    assert "Original content:\nnot json" in repair_prompt


def test_generate_with_retries_raises_after_exhausting_attempts():
    client = DummyClient([
        "still not json",
        "also not json",
    ])
    with pytest.raises(RuntimeError) as exc_info:
        summarizer._generate_with_retries(
            client=client,
            model="model",
            system="system",
            prompt="prompt",
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
    assert '"player": "Jane Doe"' in prompt
    assert '"team": "Example College"' in prompt
