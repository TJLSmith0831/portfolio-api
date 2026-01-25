import pytest
from fastapi.testclient import TestClient
from app.api.chat import CHAT_ENDPOINT
from app.main import app


@pytest.mark.integration
def test_singular_chat():
    client = TestClient(app)

    response = client.post(
        CHAT_ENDPOINT,
        json={"prompt": "Say hello in one sentence."},
    )
    print(f"Response: {response.json()}")
    assert response.status_code == 200

    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0

def test_chat_multiple_messages():
    client = TestClient(app)

    # Send first message
    response1 = client.post(
        CHAT_ENDPOINT,
        json={"prompt": "Say hello in one sentence."},
    )
    assert response1.status_code == 200

    data1 = response1.json()
    assert "response" in data1
    assert isinstance(data1["response"], str)
    assert len(data1["response"]) > 0

    # Send second message with history from the first conversation
    response2 = client.post(
        CHAT_ENDPOINT,
        json={"prompt": "Explain what I said.",
              "history": [{"role": "user", "content": "Say hello in one sentence."},
                          {"role": "assistant", "content": data1["response"]}]}
    )
    assert response2.status_code == 200

    data2 = response2.json()
    assert "response" in data2
    assert isinstance(data2["response"], str)
    assert len(data2["response"]) > 0

def test_chat_max_conversations():
    client = TestClient(app)

    history = []
    for i in range(10):
        prompt = f"Message {i+1}"
        response = client.post(
            CHAT_ENDPOINT,
            json={"prompt": prompt, "history": history}
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0

        # Add the current response to the history for the next message
        history.append({"role": "assistant", "content": data["response"]})

    # Send a message when the limit is reached
    prompt = "This should trigger the history limit reached message."
    response = client.post(
        CHAT_ENDPOINT,
        json={"prompt": prompt, "history": history}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert data["response"] == "History limit reached"
