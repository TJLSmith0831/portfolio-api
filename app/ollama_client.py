"""
Ollama client wrapper for the portfolio API.

This module provides a very small, flexible wrapper around the
official Ollama Python SDK.  The goal is to expose the SDK’s
functions without adding any type constraints that could break
when the SDK changes.
"""

import enum
import json
import os
from typing import Any

from openai import OpenAI



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

        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    # --------------------------------------------------------------------- #
    # Direct SDK passthroughs
    # --------------------------------------------------------------------- #
    def list_models(self) -> Any:
        """Return the raw response from ``client.models.list()``."""
        return self.client.models.list()

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate text with an OpenAI completion."""
        if args:
            if "model" in kwargs:
                raise TypeError("generate() received model as both positional and keyword arguments.")
            if len(args) > 1:
                raise TypeError("generate() accepts at most one positional argument.")
            kwargs["model"] = args[0]
        return self.client.completions.create(**kwargs)

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        """Run a chat‑style completion."""
        if args:
            if "model" in kwargs:
                raise TypeError("chat() received model as both positional and keyword arguments.")
            if len(args) > 1:
                raise TypeError("chat() accepts at most one positional argument.")
            kwargs["model"] = args[0]
        return self.client.chat.completions.create(**kwargs)

    def embeddings(self, *args: Any, **kwargs: Any) -> Any:
        """Generate embeddings for a prompt."""
        return ollama.embeddings(*args, **kwargs)

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def is_ollama_running(self) -> bool:
        """Check whether the Ollama server is reachable."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def chat_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        """
        Send a chat request that may include tool calls.
        Handles any returned tool_calls and feeds the result back.
        """
        response = self.chat(*args, **kwargs)

        messages = kwargs.get("messages")
        if not messages:
            return response

        choices = getattr(response, "choices", [])
        if not choices:
            return response

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if not message:
            return response

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                function = getattr(call, "function", None)
                name = getattr(function, "name", None) if function else None
                raw_arguments = getattr(function, "arguments", {}) if function else {}
                if isinstance(raw_arguments, str):
                    try:
                        arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                elif isinstance(raw_arguments, dict):
                    arguments = raw_arguments
                else:
                    arguments = {}

                result = self.call_tool(
                    name=name,
                    arguments=arguments,
                )

                messages.append({
                    "role": "tool",
                    "name": name,
                    "content": result,
                    "tool_call_id": getattr(call, "id", None),
                })

            response = self.chat(*args, **kwargs)

        return response


    def call_tool(self, name: str, arguments: dict) -> str:
        """
        Call a tool with the given name and arguments.
        """
        return "I've used a tool!"
        # if name == "get_ticket_price":
        #     city = arguments.get("destination_city")
        #     price_details = get_ticket_price(city)
        #     return json.dumps(price_details)

        # raise ValueError(f"Unknown tool: {name}")



class OllamaModels(enum.Enum):
    """
    This `OllamaModels` enum provides a set of predefined model names that
    can be used with the Ollama client. Each member of the enum is associated
    with a specific model identifier as it might be recognized by the Ollama
    server.
    """
    LLAMA='llama3.2'


def get_ollama_client() -> OllamaClient:
    """
    Get an instance of the OllamaClient.

    :return: OllamaClient An instance of the OllamaClient.
    """
    return OllamaClient()

# Add a main function to test the wrapper's functionality
if __name__ == "__main__":
    client = OllamaClient()
    print(client.is_ollama_running())
    test_prompt = "Hello, world!"
    print(f"Testing '{OllamaModels.LLAMA.value}' model with: '{test_prompt}'")
    response = client.chat(
        model=OllamaModels.LLAMA.value,
        messages=[{"role": "user", "content": test_prompt}],
    )
    print(response.choices[0].message.content)
