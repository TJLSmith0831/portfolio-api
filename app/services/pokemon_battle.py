"""
Python-driven Pokémon battle.

SirenSpec v0.1.2 has no `loop` node, so this driver invokes the
`pokemon_battle_turn` workflow once per turn and threads battle state
between calls. The narrator agent picks the attacker's move and proposes
damage; Python clamps damage to the defender's remaining HP and decides
when the battle ends.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any

from app.models.sirenspec_models import (
    BattleTurn,
    PokemonBattleRequest,
    PokemonBattleResult,
    PokemonInput,
)
from app.services.sirenspec_runner import run_workflow

log = logging.getLogger(__name__)


TYPE_MOVES: dict[str, list[str]] = {
    "fire": ["Flamethrower", "Fire Blast", "Ember", "Heat Wave"],
    "water": ["Hydro Pump", "Surf", "Bubble Beam", "Water Gun"],
    "grass": ["Solar Beam", "Vine Whip", "Razor Leaf", "Leaf Blade"],
    "electric": ["Thunderbolt", "Thunder", "Spark", "Volt Tackle"],
    "psychic": ["Psychic", "Psybeam", "Confusion", "Future Sight"],
    "normal": ["Tackle", "Body Slam", "Hyper Beam", "Quick Attack"],
    "fighting": ["Close Combat", "Karate Chop", "Dynamic Punch", "Focus Punch"],
    "rock": ["Rock Slide", "Stone Edge", "Rock Blast", "Rock Throw"],
    "ground": ["Earthquake", "Dig", "Mud Shot", "Earth Power"],
    "flying": ["Wing Attack", "Aerial Ace", "Air Slash", "Hurricane"],
    "bug": ["Bug Buzz", "X-Scissor", "Pin Missile", "Signal Beam"],
    "poison": ["Sludge Bomb", "Poison Jab", "Acid", "Venoshock"],
    "ghost": ["Shadow Ball", "Shadow Claw", "Night Shade", "Hex"],
    "ice": ["Ice Beam", "Blizzard", "Frost Breath", "Ice Shard"],
    "dragon": ["Dragon Claw", "Outrage", "Draco Meteor", "Dragon Pulse"],
    "steel": ["Iron Head", "Flash Cannon", "Metal Burst", "Steel Wing"],
}


def _allowed_moves(p: PokemonInput) -> list[str]:
    primary = p.types[0] if p.types else "normal"
    return TYPE_MOVES.get(primary, TYPE_MOVES["normal"])


def _fallback_damage(attacker: PokemonInput, defender: PokemonInput) -> int:
    base = max(5, int(attacker.attack / (defender.defense + 1) * 15) + 5)
    return max(1, int(base * (0.85 + random.random() * 0.3)))


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _parse_turn_response(text: str) -> tuple[str | None, int | None, str]:
    """
    Parse the narrator JSON response. Returns (move, damage, commentary).
    Any field may be None / empty if parsing fails.
    """
    if not text:
        return None, None, ""

    match = _JSON_OBJECT.search(text)
    blob = match.group(0) if match else text
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError:
        return None, None, text.strip()[:200]

    move = payload.get("move")
    raw_damage = payload.get("damage")
    try:
        damage = int(raw_damage) if raw_damage is not None else None
    except (TypeError, ValueError):
        damage = None
    commentary = str(payload.get("commentary") or "").strip()

    return (str(move) if move else None), damage, commentary


async def run_pokemon_battle(request: PokemonBattleRequest) -> PokemonBattleResult:
    p1 = request.pokemon1
    p2 = request.pokemon2

    hp1 = p1.hp
    hp2 = p2.hp
    attacker: int = 1 if p1.speed >= p2.speed else 2

    turns: list[BattleTurn] = []
    total_tokens = 0
    total_duration_ms = 0.0

    for turn_index in range(request.max_turns):
        if hp1 <= 0 or hp2 <= 0:
            break

        atk = p1 if attacker == 1 else p2
        defn = p2 if attacker == 1 else p1
        atk_hp = hp1 if attacker == 1 else hp2
        def_hp = hp2 if attacker == 1 else hp1

        allowed = _allowed_moves(atk)

        state = {
            "turn": turn_index + 1,
            "attacker": {
                "slot": attacker,
                "name": atk.displayName,
                "types": atk.types,
                "attack": atk.attack,
                "defense": atk.defense,
                "speed": atk.speed,
                "hp_remaining": atk_hp,
            },
            "defender": {
                "slot": 2 if attacker == 1 else 1,
                "name": defn.displayName,
                "types": defn.types,
                "attack": defn.attack,
                "defense": defn.defense,
                "speed": defn.speed,
                "hp_remaining": def_hp,
            },
            "allowed_moves": allowed,
        }

        try:
            trace = await run_workflow("pokemon_battle_turn", state)
        except Exception as exc:  # noqa: BLE001
            log.warning("pokemon turn workflow failed: %s — using fallback", exc)
            move, proposed_damage, commentary = None, None, ""
        else:
            turn_output = trace.get("output", {}).get("turn") or ""
            move, proposed_damage, commentary = _parse_turn_response(str(turn_output))

            summary = trace.get("summary") or {}
            total_tokens += int(summary.get("total_tokens") or 0)
            total_duration_ms += float(summary.get("duration_ms") or 0.0)

        if not move or move not in allowed:
            move = random.choice(allowed)
        if proposed_damage is None or proposed_damage < 1:
            proposed_damage = _fallback_damage(atk, defn)
        damage = max(1, min(int(proposed_damage), def_hp))
        if not commentary:
            commentary = f"{atk.displayName} uses {move}!"

        if attacker == 1:
            hp2 = max(0, hp2 - damage)
        else:
            hp1 = max(0, hp1 - damage)

        turns.append(
            BattleTurn(
                attacker=attacker,  # type: ignore[arg-type]
                move=move,
                damage=damage,
                commentary=commentary,
                hp1After=hp1,
                hp2After=hp2,
            )
        )

        attacker = 2 if attacker == 1 else 1

    winner: int
    if hp2 <= 0 and hp1 > 0:
        winner = 1
    elif hp1 <= 0 and hp2 > 0:
        winner = 2
    else:
        # max_turns exhausted with both still standing — the one with more HP wins.
        winner = 1 if hp1 >= hp2 else 2

    return PokemonBattleResult(
        turns=turns,
        winner=winner,  # type: ignore[arg-type]
        totalTokens=total_tokens,
        durationMs=total_duration_ms,
    )
