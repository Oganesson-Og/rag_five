import time
from typing import Optional, List
from symspellpy import SymSpell, Verbosity

# Comprehensive list based on user attachment
SHORT_CONVERSATIONAL_PHRASES: List[str] = [
    "yes", "yep", "y", "yeah", "ya", "yup", "correct", "right", "exactly",
    "no", "nope", "nah", "n", "not really", "incorrect", "wrong",
    "ok", "okay", "alright", "got it", "fine", "sounds good",
    "sure", "please", "please do", "go ahead", "go on", "carry on", "continue",
    "tell me more", "show me", "give examples", "examples", "more examples",
    "keep going", "elaborate", "expand", "clarify","make it more detailed",
    "explain further", "more details", "details", "additional details", "further",
    "next", "proceed", "move on", "what next", "and then", "may you explain it in more detail",
    "why", "how", "what about", "why so", "why is that", "how come", "how is that",
    "can you elaborate", "could you clarify", "i see", "understood", "makes sense",
    "anything else", "what else", "is that all", "any more", "go deeper", "dive deeper",
    "another example", "one more example", "what do you mean", "meaning", "define", "definition",
    "i don't understand", "confused", "lost", "repeat", "say again", "once more", "again please",
    "interesting", "good", "great", "awesome", "perfect", "excellent", "nice", "cool",
    "that's clear", "clear", "gotcha", "may you explain further"
]
# Remove duplicates by converting to set and back to list
SHORT_CONVERSATIONAL_PHRASES = sorted(list(set(phrase.lower() for phrase in SHORT_CONVERSATIONAL_PHRASES)))

# Initialize SymSpell and load dictionary
# max_dictionary_edit_distance: Maximum edit distance to create dictionary precalculations.
# prefix_length: The prefix length of dictionary words to precalculate. Max 10.
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
for phrase in SHORT_CONVERSATIONAL_PHRASES:
    sym_spell.create_dictionary_entry(phrase, 1) # count = 1, can be any positive number

SYSTEM_INVITATION_KEYWORDS: List[str] = [
    "would you like", "do you want", "shall i", "can i help", "any questions",
    "what else", "tell me more", "explain further", "show some examples",
    "anything else", "next step", "proceed?", "like me to explain", "show some examples"
]

def is_short_conversational_follow_up(query_text: str) -> bool:
    """Checks if the query is a short, common conversational follow-up, with typo correction."""
    normalized_query = query_text.lower().strip()
    if not normalized_query: # Handle empty strings
        return False

    # Lookup: get suggestions with max_edit_distance=2 (same as dictionary precalculation for efficiency)
    # Verbosity.CLOSEST returns only the closest suggestion
    suggestions = sym_spell.lookup(normalized_query, Verbosity.CLOSEST, max_edit_distance=2)
    
    if suggestions:
        corrected_query = suggestions[0].term
        return corrected_query in SHORT_CONVERSATIONAL_PHRASES
    else:
        # If no suggestion is found (e.g., query is too different or empty after strip),
        # fall back to checking the original normalized query.
        # This might be useful if the phrase is correct but not in the dictionary for some reason,
        # though with SymSpell loading all phrases, this fallback is less critical for correctness.
        return normalized_query in SHORT_CONVERSATIONAL_PHRASES

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