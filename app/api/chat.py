"""
The chat module provides endpoints for interacting with the Ollama-backed LLM.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.models.chat_model import ChatRequest
from app.llm_client import LLMClient, OllamaModels, get_llm_client

CHAT_ENDPOINT = "/chat"

router = APIRouter(tags=["Chat"])

@router.post("/chat")
def chat(request: ChatRequest, client: LLMClient = Depends(get_llm_client)) -> StreamingResponse:
    """
    Endpoint for interacting with the Ollama-backed LLM.

    :param request: The chat request containing the user's prompt and conversation history.
    :param client: The Ollama client instance.
    :return: The chat response containing the LLM's response.
    """

    if len(request.history) >= 10:
        return StreamingResponse(
            iter(["History limit reached"]),
            media_type="text/plain",
        )

    request.history.append({"role": "user", "content": request.prompt})

    def token_generator():
        try:
            stream = client.chat(
                model=client.model_enum.value,
                messages=request.history,
                stream=True,
            )

            full_content = ""

            for chunk in stream:
                delta = (
                    getattr(chunk.choices[0].delta, "content", None)
                    if hasattr(chunk.choices[0], "delta")
                    else getattr(chunk, "response", None)
                )
                if delta:
                    full_content += delta
                    yield delta

            request.history.append(
                {"role": "assistant", "content": full_content}
            )

        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return StreamingResponse(token_generator(), media_type="text/plain")
