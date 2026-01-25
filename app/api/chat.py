"""
The chat module provides endpoints for interacting with the Ollama-backed LLM.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from app.ollama_client import OllamaClient, OllamaModels

CHAT_ENDPOINT = "/api/chat/"

router = APIRouter(tags=["Chat"])

def get_ollama_client() -> OllamaClient:
    return OllamaClient()

class ChatRequest(BaseModel):
    prompt: str = Field(
        description="Prompt to send to the LLM"
    )
    history: list[dict[str, str]] = Field(
        default=[],
        description="History of previous interactions"
    )

class ChatResponse(BaseModel):
    response: str = Field(
        description="Model-generated response"
    )

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

        # Add the LLM response to the history
        request.history.append({"role": "assistant", "content": result["message"]["content"]})

        return ChatResponse(
            response=result["message"]["content"]
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
