"""
Unit tests for the Pokémon battle Python loop driver.

The SirenSpec turn workflow is stubbed via monkey-patching so these
tests run with no API keys.
"""

import pytest

from app.models.sirenspec_models import PokemonBattleRequest, PokemonInput
from app.services import pokemon_battle as pb


def _mk(name: str, hp: int, attack: int, defense: int, speed: int, types: list[str]):
    return PokemonInput(
        name=name.lower(),
        displayName=name,
        hp=hp,
        attack=attack,
        defense=defense,
        speed=speed,
        types=types,
    )


def test_parse_turn_response_clean_json():
    text = '{"move": "Thunderbolt", "damage": 17, "commentary": "Electrifying!"}'
    move, damage, commentary = pb._parse_turn_response(text)
    assert move == "Thunderbolt"
    assert damage == 17
    assert commentary == "Electrifying!"


def test_parse_turn_response_extracts_json_from_prose():
    text = 'Sure! Here you go: {"move":"Tackle","damage":5,"commentary":"Smack."} Hope that helps.'
    move, damage, commentary = pb._parse_turn_response(text)
    assert move == "Tackle"
    assert damage == 5
    assert commentary == "Smack."


def test_parse_turn_response_unparseable_returns_text_fallback():
    move, damage, commentary = pb._parse_turn_response("model went off-script")
    assert move is None
    assert damage is None
    assert "off-script" in commentary


@pytest.mark.asyncio
async def test_battle_loop_terminates_when_one_faints(monkeypatch):
    p1 = _mk("Pikachu", hp=35, attack=55, defense=40, speed=90, types=["electric"])
    p2 = _mk("Caterpie", hp=10, attack=10, defense=10, speed=20, types=["bug"])

    async def fake_workflow(_name, _state):
        return {
            "output": {
                "turn": '{"move":"Thunderbolt","damage":50,"commentary":"Zap!"}'
            },
            "summary": {"total_tokens": 100, "duration_ms": 10.0},
        }

    monkeypatch.setattr(pb, "run_workflow", fake_workflow)

    req = PokemonBattleRequest(pokemon1=p1, pokemon2=p2, max_turns=10)
    result = await pb.run_pokemon_battle(req)

    # Pikachu faster, one-shots Caterpie (damage clamped to 10).
    assert len(result.turns) == 1
    assert result.winner == 1
    assert result.turns[0].damage == 10
    assert result.turns[0].hp2After == 0
    assert result.totalTokens == 100


@pytest.mark.asyncio
async def test_battle_loop_uses_fallback_when_model_invalid(monkeypatch):
    p1 = _mk("Bulbasaur", hp=45, attack=49, defense=49, speed=45, types=["grass"])
    p2 = _mk("Squirtle", hp=44, attack=48, defense=65, speed=43, types=["water"])

    async def fake_workflow(_name, _state):
        # Bogus move name — driver should swap to a valid allowed move.
        return {
            "output": {
                "turn": '{"move":"Not A Real Move","damage":0,"commentary":""}'
            },
            "summary": {},
        }

    monkeypatch.setattr(pb, "run_workflow", fake_workflow)

    req = PokemonBattleRequest(pokemon1=p1, pokemon2=p2, max_turns=2)
    result = await pb.run_pokemon_battle(req)

    assert len(result.turns) == 2
    for turn in result.turns:
        # Driver replaced the bogus move with a real grass / water move.
        assert turn.move in (
            pb.TYPE_MOVES["grass"] + pb.TYPE_MOVES["water"]
        )
        assert turn.damage >= 1
        assert turn.commentary  # filled in by fallback


@pytest.mark.asyncio
async def test_battle_speed_decides_first_attacker(monkeypatch):
    p1 = _mk("Slow", hp=100, attack=20, defense=20, speed=10, types=["normal"])
    p2 = _mk("Fast", hp=100, attack=20, defense=20, speed=99, types=["normal"])

    async def fake_workflow(_name, _state):
        return {
            "output": {"turn": '{"move":"Tackle","damage":5,"commentary":"!"}'},
            "summary": {},
        }

    monkeypatch.setattr(pb, "run_workflow", fake_workflow)

    req = PokemonBattleRequest(pokemon1=p1, pokemon2=p2, max_turns=1)
    result = await pb.run_pokemon_battle(req)

    # p2 (Fast) attacks first.
    assert result.turns[0].attacker == 2
