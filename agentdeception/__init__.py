"""AgentDeception - LLM evaluation for covert action and deception."""

from agentdeception.task import (
    deception_neutral,
    deception_critical_only,
    deception_full,
    deception_cot,
    deception_critical_cot,
)

__all__ = [
    "deception_neutral",
    "deception_critical_only",
    "deception_full",
    "deception_cot",
    "deception_critical_cot",
]
