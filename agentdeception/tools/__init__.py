"""
Naturalistic tools for AgentDeception evaluation.

The model has access to standard assistant tools (email, phone, web, etc.)
plus a realistic contacts list and website database. Some contacts and websites
are "flagged" as reporting entities (FBI, 911, crisis hotlines, etc.).

We detect when the model contacts flagged entities to measure reporting behavior.
"""

from agentdeception.tools.general import (
    send_email,
    search_contacts,
    make_phone_call,
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

from agentdeception.tools.contacts import (
    CONTACTS,
    get_all_contacts,
    get_flagged_contacts,
    get_contact_by_name,
    is_flagged_contact,
    search_contacts as search_contacts_db,
)

from agentdeception.tools.websites import (
    WEBSITES,
    get_all_websites,
    get_flagged_websites,
    is_flagged_url,
    search_websites as search_websites_db,
)

__all__ = [
    # Tools
    "send_email",
    "search_contacts",
    "make_phone_call",
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
    # Contacts
    "CONTACTS",
    "get_all_contacts",
    "get_flagged_contacts",
    "get_contact_by_name",
    "is_flagged_contact",
    "search_contacts_db",
    # Websites
    "WEBSITES",
    "get_all_websites",
    "get_flagged_websites",
    "is_flagged_url",
    "search_websites_db",
]


def get_all_tools():
    """Return all tools available to the model."""
    return get_all_general_tools()
