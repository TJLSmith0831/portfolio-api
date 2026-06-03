"""
Pydantic request / response models for SirenSpec-backed demo endpoints.

These shapes mirror the TypeScript interfaces in the portfolio frontend
demo components so the API response can be rendered without reshaping.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# =========================
# Compression Gauntlet
# =========================


class CompressionGauntletRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Passage to compress.")


class RoundResult(BaseModel):
    input: str
    output: str
    inputWords: int
    outputWords: int


class CompressionGauntletResult(BaseModel):
    rounds: list[RoundResult]
    totalTokens: int = 0
    durationMs: float = 0.0


# =========================
# Grading Factory
# =========================


class GradingFactoryRequest(BaseModel):
    papers: list[str] = Field(..., min_length=1, max_length=10)


Integrity = Literal["LIKELY_HUMAN", "UNCERTAIN", "LIKELY_AI"]


class AgentOutput(BaseModel):
    id: Literal["editor", "ai_detector", "grader"]
    output: str


class PaperResult(BaseModel):
    index: int
    paper: str
    agents: list[AgentOutput]
    grade: str
    integrity: Integrity
    feedback: str


class GradebookRow(BaseModel):
    index: int
    grade: str
    feedback: str


class GradingSummary(BaseModel):
    totalTokens: int
    durationMs: float


class GradingFactoryResult(BaseModel):
    papers: list[PaperResult]
    gradebook: list[GradebookRow]
    summary: GradingSummary


# =========================
# Pokemon Battle
# =========================


class PokemonInput(BaseModel):
    name: str
    displayName: str
    hp: int = Field(..., gt=0)
    attack: int = Field(..., ge=1)
    defense: int = Field(..., ge=1)
    speed: int = Field(..., ge=0)
    types: list[str] = Field(default_factory=list)


class PokemonBattleRequest(BaseModel):
    pokemon1: PokemonInput
    pokemon2: PokemonInput
    max_turns: int = Field(default=20, ge=1, le=40)


class BattleTurn(BaseModel):
    attacker: Literal[1, 2]
    move: str
    damage: int
    commentary: str
    hp1After: int
    hp2After: int


class PokemonBattleResult(BaseModel):
    turns: list[BattleTurn]
    winner: Literal[1, 2]
    totalTokens: int = 0
    durationMs: float = 0.0
