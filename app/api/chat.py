"""
The chat module provides endpoints for interacting with the Ollama-backed LLM.
"""
from fastapi import APIRouter, Depends, HTTPException
from app.models.chat_model import ChatRequest, ChatResponse
from app.ollama_client import OllamaClient, OllamaModels, get_ollama_client

CHAT_ENDPOINT = "/api/chat/"

router = APIRouter(tags=["Chat"])

@router.post("/chat/", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    client: OllamaClient = Depends(get_ollama_client),
) -> ChatResponse:
    try:
        # Ensure the history does not exceed 10 messages
        if len(request.history) >= 10:
            return ChatResponse(response="History limit reached")

        # Add the new user message to the history
        request.history.append({"role": "user", "content": request.prompt})

        # Call the Ollama client with the updated conversation history
        result = client.chat(
            model=OllamaModels.LLAMA.value,
            messages=request.history
        )

        content = result.choices[0].message.content

        # Add the LLM response to the history
        request.history.append({"role": "assistant", "content": content})

        return ChatResponse(
            response=content
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
