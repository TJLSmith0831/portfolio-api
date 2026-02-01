
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """Chat request."""
    prompt: str = Field(
        description="Prompt to send to the LLM"
    )
    history: list[dict[str, str]] = Field(
        default=[],
        description="History of previous interactions"
    )

class ChatResponse(BaseModel):
    """Chat response."""
    response: str = Field(
        description="Model-generated response"
    )
