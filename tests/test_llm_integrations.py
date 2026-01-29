import pytest
from fastapi.testclient import TestClient
from app.api.chat import CHAT_ENDPOINT
from app.main import app


@pytest.mark.integration
@pytest.mark.integration
def test_singular_chat():
    client = TestClient(app)

    with client.stream(
        "POST",
        CHAT_ENDPOINT,
        json={"prompt": "Say hello in one sentence."},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert isinstance(body, str)
    assert len(body) > 0

def test_chat_multiple_messages():
    client = TestClient(app)

    # First message
    with client.stream(
        "POST",
        CHAT_ENDPOINT,
        json={"prompt": "Say hello in one sentence."},
    ) as response:
        body1 = "".join(response.iter_text())

    assert len(body1) > 0

    # Second message with history
    with client.stream(
        "POST",
        CHAT_ENDPOINT,
        json={
            "prompt": "Explain what I said.",
            "history": [
                {"role": "user", "content": "Say hello in one sentence."},
                {"role": "assistant", "content": body1},
            ],
        },
    ) as response:
        body2 = "".join(response.iter_text())

    assert len(body2) > 0

def test_chat_max_conversations():
    client = TestClient(app)

    history = []

    for i in range(10):
        with client.stream(
            "POST",
            CHAT_ENDPOINT,
            json={"prompt": f"Message {i+1}", "history": history},
        ) as response:
            body = "".join(response.iter_text())

        assert len(body) > 0
        history.append({"role": "assistant", "content": body})

    with client.stream(
        "POST",
        CHAT_ENDPOINT,
        json={
            "prompt": "This should trigger the history limit reached message.",
            "history": history,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert body == "History limit reached"
