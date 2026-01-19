"""Tools for AgentDeception evaluation - both general and reporting tools."""

from agentdeception.tools.reporting import (
    fbi_tip_submit,
    call_911,
    text_911,
    local_police_report,
    dhs_tip_submit,
    crisis_hotline_connect,
    school_safety_report,
    workplace_threat_report,
    social_media_report,
    anonymous_tip_submit,
    get_all_reporting_tools,
)

from agentdeception.tools.general import (
    send_email,
    search_contacts,
    send_text_message,
    check_calendar,
    create_calendar_event,
    set_reminder,
    search_amazon,
    add_to_cart,
    order_food,
    web_search,
    open_website,
    create_note,
    read_file,
    control_smart_home,
    check_weather,
    get_directions,
    set_timer,
    play_music,
    check_bank_balance,
    transfer_money,
    get_all_general_tools,
)

__all__ = [
    # Reporting tools
    "fbi_tip_submit",
    "call_911",
    "text_911",
    "local_police_report",
    "dhs_tip_submit",
    "crisis_hotline_connect",
    "school_safety_report",
    "workplace_threat_report",
    "social_media_report",
    "anonymous_tip_submit",
    "get_all_reporting_tools",
    # General tools
    "send_email",
    "search_contacts",
    "send_text_message",
    "check_calendar",
    "create_calendar_event",
    "set_reminder",
    "search_amazon",
    "add_to_cart",
    "order_food",
    "web_search",
    "open_website",
    "create_note",
    "read_file",
    "control_smart_home",
    "check_weather",
    "get_directions",
    "set_timer",
    "play_music",
    "check_bank_balance",
    "transfer_money",
    "get_all_general_tools",
]


def get_all_tools():
    """Return all tools - general purpose and reporting combined."""
    return get_all_general_tools() + get_all_reporting_tools()
