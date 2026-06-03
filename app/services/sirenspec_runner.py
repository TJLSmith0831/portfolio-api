"""
Thin async wrapper around SirenSpec's load + execute primitives.

Workflow YAMLs live under app/workflows/. The runner resolves names to
absolute paths so callers can stay short and so tests can override the
workflow directory via the WORKFLOWS_DIR env var.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sirenspec import execute, load_workflow

_DEFAULT_WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"


def workflows_dir() -> Path:
    """Return the directory holding workflow YAML files."""
    override = os.getenv("WORKFLOWS_DIR")
    if override:
        return Path(override)
    return _DEFAULT_WORKFLOWS_DIR


def workflow_path(name: str) -> Path:
    """Resolve a workflow file name to its absolute path."""
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    return workflows_dir() / name


async def run_workflow(name: str, user_input: str | list | dict) -> dict[str, Any]:
    """
    Execute a workflow by name and return SirenSpec's structured trace.

    :param name: Workflow file stem under app/workflows/ (e.g. "compression_gauntlet").
    :param user_input: The initial user message. Lists / dicts are JSON-encoded
        so factory `for_each: "{{ inputs.message }}"` can parse them.
    :return: The SirenSpec trace dict with top-level keys: workflow, input,
        nodes, output, summary.
    """
    if not isinstance(user_input, str):
        user_input = json.dumps(user_input)

    workflow = load_workflow(str(workflow_path(name)))
    return await execute(workflow, user_input=user_input)


def node_trace(trace: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    """Find a single node's trace entry by id; None if missing."""
    for n in trace.get("nodes", []):
        if n.get("id") == node_id:
            return n
    return None
