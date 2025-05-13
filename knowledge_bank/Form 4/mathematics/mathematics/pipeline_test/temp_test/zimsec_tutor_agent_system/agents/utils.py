import time
from typing import Optional, List

SHORT_CONVERSATIONAL_PHRASES: List[str] = [
    "yes", "yep", "y", "no", "nope", "n", "ok", "okay", "sure", "please",
    "go on", "tell me more", "examples", "more examples", "continue",
    "explain further", "more details", "next", "proceed", "yup",
    "why", "how", "what about", "please continue"
]

SYSTEM_INVITATION_KEYWORDS: List[str] = [
    "would you like", "do you want", "shall i", "can i help", "any questions",
    "what else", "tell me more", "explain further", "show some examples",
    "anything else", "next step", "proceed?", "like me to explain", "show some examples"
]

def is_short_conversational_follow_up(query_text: str) -> bool:
    """Checks if the query is a short, common conversational follow-up."""
    return query_text.lower().strip() in SHORT_CONVERSATIONAL_PHRASES

def did_system_invite_follow_up(system_response_text: Optional[str]) -> bool:
    """Checks if the system's last response likely invited a follow-up."""
    if not system_response_text:
        return False
    lower_response = system_response_text.lower()
    return any(kw in lower_response for kw in SYSTEM_INVITATION_KEYWORDS)

def is_conversation_stale(interaction_timestamp: Optional[float], stale_threshold_seconds: int) -> bool:
    """Checks if the conversation is stale based on the last interaction time."""
    if interaction_timestamp is None:
        return True  # No previous interaction, so it's "stale" for a new context
    return (time.time() - interaction_timestamp) > stale_threshold_seconds

# Placeholder for estimate_tokens if it were to be moved here.
# For now, it seems to be used by other agents or not at all in orchestrator.
# def estimate_tokens(text: str) -> int:
#     """Estimates the number of tokens in a string.
#     A simple approximation: 1 token ~ 4 chars in English.
#     Or, for a more common one, average characters per token is often around 3-4 for English text.
#     This is a very rough heuristic.
#     """
#     return len(text) // 4 