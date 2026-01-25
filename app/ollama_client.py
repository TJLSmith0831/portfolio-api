"""
Ollama client wrapper for the portfolio API.

This module provides a very small, flexible wrapper around the
official Ollama Python SDK.  The goal is to expose the SDK’s
functions without adding any type constraints that could break
when the SDK changes.
"""

from dataclasses import dataclass
import enum
import os
from typing import Any

import ollama

class OllamaClient:
    """
    Thin wrapper around the Ollama Python SDK.

    The wrapper simply forwards calls to the SDK, allowing the
    rest of the application to remain agnostic of the SDK’s
    concrete API shape.
    """

    def __init__(self, host: str | None = None) -> None:
        """
        Configure the SDK host.

        :param host: Optional Ollama host URL (e.g., "http://localhost:11434").
                     If omitted, the SDK uses its default.
        """
        if host:
            # The SDK reads the host from the ``OLLAMA_HOST`` environment
            # variable, so we set it here.
            os.environ["OLLAMA_HOST"] = host

    # --------------------------------------------------------------------- #
    # Direct SDK passthroughs
    # --------------------------------------------------------------------- #
    def list_models(self) -> Any:
        """Return the raw response from ``ollama.list()``."""
        return ollama.list()

    def pull_model(self, model_name: str) -> Any:
        """Pull a model from the Ollama registry."""
        return ollama.pull(model_name)

    def delete_model(self, model_name: str) -> Any:
        """Delete a local Ollama model."""
        return ollama.delete(model_name)

    def show_model(self, model_name: str) -> Any:
        """Retrieve detailed information about a model."""
        return ollama.show(model_name)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate text with an Ollama model."""
        return ollama.generate(*args, **kwargs)

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        """Run a chat‑style completion."""
        return ollama.chat(*args, **kwargs)

    def embeddings(self, *args: Any, **kwargs: Any) -> Any:
        """Generate embeddings for a prompt."""
        return ollama.embeddings(*args, **kwargs)

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def is_ollama_running(self) -> bool:
        """Check whether the Ollama server is reachable."""
        try:
            ollama.list()
            return True
        except Exception:
            return False

class OllamaModels(enum.Enum):
    """
    This `OllamaModels` enum provides a set of predefined model names that
    can be used with the Ollama client. Each member of the enum is associated
    with a specific model identifier as it might be recognized by the Ollama
    server.
    """
    LLAMA='llama3.2'

# Add a main function to test the wrapper's functionality
if __name__ == "__main__":
    client = OllamaClient()
    print(client.is_ollama_running())
    test_prompt="Hello, world!"
    print(f"Testing '{OllamaModels.LLAMA.value}' model with: '{test_prompt}'")
    response = client.generate(OllamaModels.LLAMA.value, prompt=test_prompt)
    print(response.response)
