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
import threading
from typing import Any

from openai import OpenAI

class OllamaModels(enum.Enum):
    """
    Enum of predefined Ollama model names.

    Each member corresponds to a model identifier used by the Ollama server.
    """
    GEMMA4_31B='gemma4:31b'


class LLMClient:
    """
    Thin wrapper around the Ollama Python SDK.

    This class forwards calls to the Ollama SDK client,
    abstracting the underlying SDK interface.

    :param model: Ollama model to use (default is OllamaModels.LLAMA_1B)
    :param host: Optional host URL for the Ollama server
    """

    def __init__(self, model: OllamaModels = OllamaModels.GEMMA4_31B, host: str | None = None) -> None:
        """
        Initialize the LLM client with optional model and host.

        :param model: Model enum to use for LLM requests.
        :param host: Optional Ollama host URL to override environment setting.
        """
        if host:
            # The SDK reads the host from the ``OLLAMA_HOST`` environment
            # variable, so we set it here.
            os.environ["OLLAMA_HOST"] = host

        base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1")
        api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model_enum = model
        self.model_name = model.value

    # --------------------------------------------------------------------- #
    # Direct SDK passthroughs
    # --------------------------------------------------------------------- #
    def list_models(self) -> Any:
        """
        List available LLM models from the Ollama server.

        :return: Raw response from the SDK models list call.
        """
        return self.client.models.list()

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        """
        Create a chat completion from the LLM.

        Accepts either one positional argument as the model name or model keyword argument.

        :return: Chat completion response object.
        """
        if len(args) > 0:
            if "model" in kwargs:
                raise TypeError("chat() received model as both positional and keyword arguments.")
            if len(args) > 1:
                raise TypeError("chat() accepts at most one positional argument.")
            kwargs["model"] = args[0]
        return self.client.chat.completions.create(**kwargs)

    def embeddings(self, *args: Any, **kwargs: Any) -> Any:
        """
        Generate text embeddings.

        :return: Embeddings response object.
        """
        return self.client.embeddings.create(*args, **kwargs)

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def is_ollama_running(self) -> bool:
        """
        Check if the Ollama backend server is running and reachable.

        :return: True if server responds, False otherwise.
        """
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def chat_with_tools(self, *args: Any, **kwargs: Any) -> Any:
        """
        Perform a chat completion that may include tool calls.

        This method handles invoking any tools requested by the LLM
        then continues the chat with tool results included.

        :return: The updated chat completion including tool results.
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

    def call_tool(self, name: Any, arguments: dict) -> str:
        """
        Call a tool identified by name with provided arguments.

        :param name: Tool name to invoke
        :param arguments: Arguments dictionary for the tool call
        :return: Tool output as a string
        """
        return "I've used a tool!"
        # if name == "get_ticket_price":
        #     city = arguments.get("destination_city")
        #     price_details = get_ticket_price(city)
        #     return json.dumps(price_details)

        # raise ValueError(f"Unknown tool: {name}")

# ===================
# LLM Singleton
# ===================
_client: LLMClient | None = None
_client_lock = threading.Lock()

def get_llm_client() -> LLMClient:
    """
    Return a process-wide singleton LLMClient.

    This function ensures that there is only one instance
    of LLMClient in the process, initialized on demand.

    :return: Singleton instance of LLMClient
    """
    global _client

    if _client is None:
        with _client_lock:
            if _client is None:
                model_name = os.getenv("OLLAMA_MODEL", "GEMMA4_31B")

                if model_name:
                    try:
                        model_enum = OllamaModels[model_name]
                    except KeyError as exc:
                        raise ValueError(
                            f"Invalid OLLAMA_MODEL '{model_name}'. "
                            f"Valid values: {[m.name for m in OllamaModels]}"
                        ) from exc
                else:
                    model_enum = OllamaModels.GEMMA4_31B

                _client = LLMClient(model=model_enum)

    return _client


# Add a main function to test the wrapper's functionality
if __name__ == "__main__":
    client = LLMClient()
    print(client.is_ollama_running())
    test_prompt = "Hello, world!"
    print(f"Testing '{OllamaModels.GEMMA4_31B.value}' model with: '{test_prompt}'")
    response = client.chat(
        model=OllamaModels.GEMMA4_31B.value,
        messages=[{"role": "user", "content": test_prompt}],
    )
    print(response.choices[0].message.content)
