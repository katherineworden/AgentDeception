"""
Prompt dataset definitions and utilities.

This module loads prompts from ReportBench for use in the deception eval.
Prompts are categorized by:
1. Category (suicide, violence, terrorism, etc.)
2. Severity (1-5 scale from mild/ambiguous to explicit/imminent threat)
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any
from pydantic import BaseModel


class Severity(int, Enum):
    """
    Severity levels for evaluation prompts.

    Level 1: Mild/Ambiguous - Could be benign, vague expressions
    Level 2: Concerning - Shows signs of distress but no clear intent
    Level 3: Moderate - Clear distress or concerning interest, no specific plans
    Level 4: Serious - Specific intent expressed, may have partial plans
    Level 5: Critical - Explicit imminent threat with specific plans/means
    """

    MILD = 1
    CONCERNING = 2
    MODERATE = 3
    SERIOUS = 4
    CRITICAL = 5


class PromptCategory(str, Enum):
    """Categories of concerning content for evaluation."""

    SUICIDE_SELF_HARM = "suicide_self_harm"
    VIOLENCE_HOMICIDE = "violence_homicide"
    TERRORISM = "terrorism"
    CHILD_SAFETY = "child_safety"
    WEAPONS_EXPLOSIVES = "weapons_explosives"
    BIOWEAPONS_GOF = "bioweapons_gof"
    MASS_CASUALTY = "mass_casualty"
    DOMESTIC_VIOLENCE = "domestic_violence"
    SCHOOL_THREATS = "school_threats"
    RADICALIZATION = "radicalization"


class EvalPrompt(BaseModel):
    """A single evaluation prompt with metadata."""

    id: str
    category: PromptCategory
    severity: Severity
    prompt: str
    context: str | None = None
    expected_response_type: str | None = None
    notes: str | None = None


def load_prompts(filepath: Path | None = None) -> list[EvalPrompt]:
    """
    Load prompts from data/prompts/*.json files.

    Args:
        filepath: Optional path to specific JSON file

    Returns:
        List of EvalPrompt objects
    """
    prompts = []

    # Find the data directory
    current_file = Path(__file__)
    data_dir = current_file.parent.parent.parent / "data" / "prompts"

    if filepath and filepath.exists():
        # Load specific file
        with open(filepath) as f:
            data = json.load(f)
            prompts.extend([EvalPrompt(**p) for p in data])
    elif data_dir.exists():
        # Load all JSON files from data/prompts
        for json_file in sorted(data_dir.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
                prompts.extend([EvalPrompt(**p) for p in data])

    return prompts


def get_prompts_by_severity(
    prompts: list[EvalPrompt], severity: Severity
) -> list[EvalPrompt]:
    """Filter prompts by severity level."""
    return [p for p in prompts if p.severity == severity]


def get_prompts_by_category(
    prompts: list[EvalPrompt], category: PromptCategory
) -> list[EvalPrompt]:
    """Filter prompts by category."""
    return [p for p in prompts if p.category == category]
