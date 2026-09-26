"""Small Gemini runtime used by the Red and Red Advance lab agents.

It mirrors the OpenAI-compatible runner while avoiding ADK workflow retries that
can turn a transient 503 into a long-running checkpoint process.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Callable

from google import genai
from google.genai import types


@dataclass
class GeminiAgent:
    name: str
    instruction: str
    provider: str = "gemini"


@dataclass
class GeminiRunner:
    app_name: str
    model: str
    provider: str = "gemini"
    temperature: float = 0.4
    input_hooks: list[Callable[[str], str | None]] = field(default_factory=list)
    output_hooks: list[Callable[[str], str]] = field(default_factory=list)

    async def chat(self, agent: GeminiAgent, user_message: str) -> str:
        for hook in self.input_hooks:
            blocked = hook(user_message)
            if blocked:
                return blocked

        def generate() -> str:
            client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
            response = client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=agent.instruction,
                    temperature=self.temperature,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            return (response.text or "").strip()

        text = await asyncio.to_thread(generate)
        for hook in self.output_hooks:
            text = hook(text)
        return text


def create_gemini_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    model: str,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[GeminiAgent, GeminiRunner]:
    return (
        GeminiAgent(name=name, instruction=instruction),
        GeminiRunner(
            app_name=app_name,
            model=model,
            input_hooks=list(input_hooks or []),
            output_hooks=list(output_hooks or []),
            temperature=temperature,
        ),
    )
