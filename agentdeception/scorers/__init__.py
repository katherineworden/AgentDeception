"""Scorers for AgentDeception evaluation."""

from agentdeception.scorers.tool_usage import (
    get_tools_used,
    get_reporting_actions,
    get_max_escalation_level,
    did_report,
    ESCALATION_LEVELS,
    ALL_REPORTING_TOOLS,  # Empty set for backwards compatibility
)

__all__ = [
    "get_tools_used",
    "get_reporting_actions",
    "get_max_escalation_level",
    "did_report",
    "ESCALATION_LEVELS",
    "ALL_REPORTING_TOOLS",
]
