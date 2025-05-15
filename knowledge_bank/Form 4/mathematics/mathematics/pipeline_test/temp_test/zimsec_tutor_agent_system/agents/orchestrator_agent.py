# orchestrator_agent.py
"""
ZIMSEC Tutoring System - Orchestrator Agent
-------------------------------------------

This module defines the `OrchestratorAgent`, the central coordinating agent in
the ZIMSEC Tutoring System. It manages the conversation flow, determines user
intent, interacts with the `CurriculumAlignmentAgent`, and routes tasks to
specialist agents (e.g., `ConceptTutorAgent`, `AssessmentRevisionAgent`).

Key Responsibilities:
- Receives user queries from the `StudentInterfaceAgent` (via `UserProxy`).
- Manages conversation context, including recent alignment data and interaction history.
- Determines if a query is a follow-up or a new topic using an LLM-based intent classifier
  and utility functions (`is_short_conversational_follow_up`, `did_system_invite_follow_up`, `is_conversation_stale`).
- If necessary, consults the `CurriculumAlignmentAgent` to get syllabus alignment for the query.
- Based on alignment and intent, routes the query to the appropriate specialist agent:
    - `ConceptTutorAgent` for explanations and conceptual help.
    - `DiagnosticRemediationAgent` for checking answers and diagnosing misconceptions.
    - `AssessmentRevisionAgent` for practice questions and revision tasks.
    - `ProjectsMentorAgent` for CALA project assistance.
    - `ContentGenerationAgent` for generating learning materials.
    - `AnalyticsProgressAgent` for progress tracking and analytics queries.
- Handles cases where queries are out of syllabus scope or require subject clarification.
- Forwards the specialist agent's response (or its own direct response) back to the `StudentInterfaceAgent`.
- Manages state for multi-turn interactions, such as subject clarification dialogues.

Technical Details:
- Inherits from `autogen.AssistantAgent`.
- Defines a comprehensive system message detailing its roles, communication protocols with other agents, and decision-making logic.
- Registers a primary reply function (`_generate_orchestrator_reply`) to handle incoming messages.
- Uses an internal LLM (`_classify_intent_with_llm`) for nuanced intent classification.
- Interacts with other agents by initiating chats and processing their JSON responses.
- Maintains `current_session_alignment` to store the latest syllabus alignment data.
- Includes logic for handling simulated user input for context-switching clarifications (due to limitations in async human input with UserProxyAgent in some Autogen setups).

Dependencies:
- autogen
- json
- asyncio
- typing
- os
- time
- logging
- datetime
- re
- .utils (helper functions for conversation context analysis)
- autogen_core.models (SystemMessage, UserMessage)
- Other agents in the system (CurriculumAlignmentAgent, ConceptTutorAgent, etc.)

Author: Keith Satuku
Version: 1.0.0
Created: 2024
License: MIT
"""
# NOTE: Due to issues with handling async human input via UserProxyAgent
# in the current setup (causing TypeError: object str can't be used in 'await' expression),
# the logic to handle the user's reply to context-switch clarifications is currently
# SIMULATED within the _generate_orchestrator_reply function itself.
# Search for 'SIMULATION START' and 'SIMULATION END' comments below.
# This allows testing the subsequent logic flow without needing live user input.
# In a production system, the state management and reply handling would need
# to properly integrate with the UserProxyAgent's input mechanism.

import autogen
import json
import asyncio # Added for async operations in reply and testing
from typing import List, Dict, Any, Optional, Tuple, Union # Still needed for other type hints potentially
import os
import time
import logging
from datetime import datetime
import re
from .utils import ( # Assuming utils.py is in the same directory
    is_short_conversational_follow_up,
    did_system_invite_follow_up,
    is_conversation_stale,
    # estimate_tokens # Not used directly in this snippet anymore
)
from autogen_core.models import SystemMessage, UserMessage # Added
# from autogen_ext.models.openai import OpenAIChatCompletionClient # No longer directly instantiating this here

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Conversation constants
STALE_THRESHOLD_SECONDS = 3600 # 1 hour

# Removed SHORT_CONVERSATIONAL_PHRASES and the following function definitions 
# as they are now imported from .utils:
# - is_short_conversational_follow_up
# - did_system_invite_follow_up
# - is_conversation_stale

# Helper function to estimate tokens (rough approximation)
def estimate_tokens(text: str) -> int:
    """Rough estimate of tokens based on characters and words.

    Provides a basic heuristic to approximate the number of tokens in a given text.
    This is a simplified estimation and may not accurately reflect the token count
    of specific LLM tokenizers.

    Args:
        text (str): The input string.

    Returns:
        int: An estimated token count.
    """
    # Average of 4 chars per token and 1.3 tokens per word
    char_estimate = len(text) / 4
    word_estimate = len(text.split()) * 1.3
    return int((char_estimate + word_estimate) / 2)

class OrchestratorAgent(autogen.AssistantAgent):
    """
    The OrchestratorAgent acts as the central coordinator in the multi-agent tutoring system.

    It receives user queries, determines their context and intent, aligns them with the
    curriculum by consulting the `CurriculumAlignmentAgent`, and then routes them to
    the most appropriate specialist agent for handling. It manages the overall flow
    of conversation and ensures that responses are relevant and syllabus-aligned.

    Attributes:
        curriculum_alignment_agent (CurriculumAlignmentAgent): Agent for syllabus alignment.
        concept_tutor_agent (ConceptTutorAgent): Agent for conceptual explanations.
        diagnostic_agent (DiagnosticRemediationAgent): Agent for diagnosing issues.
        assessment_agent (AssessmentRevisionAgent): Agent for assessments.
        projects_mentor_agent (ProjectsMentorAgent): Agent for project mentorship.
        content_generation_agent (ContentGenerationAgent): Agent for content creation.
        analytics_progress_agent (AnalyticsProgressAgent): Agent for tracking progress.
        current_session_alignment (Optional[Dict]): Stores the latest syllabus alignment data for the session.
        _active_sub_chat_partner_name (Optional[str]): Name of the agent currently in a sub-chat with the orchestrator.
        stale_threshold_seconds (int): Time in seconds after which conversation context is considered stale.
    """
    def __init__(self, name, llm_config, curriculum_alignment_agent, concept_tutor_agent, diagnostic_agent, assessment_agent, projects_mentor_agent, content_generation_agent, analytics_progress_agent, **kwargs):
        system_message = (
    "You are the Orchestrator Agent in a multi-agent AI tutoring system for ZIMSEC O-Level students.\n"
    "Your primary roles are:\n"
    "1. Receive the initial user query from the UserProxy.\n"
    "2. Determine if the query is a direct follow-up to a recent interaction. If so, and the context is clear and recent, reuse existing alignment. \n"
    "3. If the query is not a clear follow-up, or if the context is stale, consult the CurriculumAlignmentAgent. You will send it a JSON message like: {\"user_query\": \"<actual_user_query>\", \"learner_context\": {\"current_subject_hint\": \"<current_subject_or_default>\"}}.\n"
    "4. Receive and parse the JSON response from the CurriculumAlignmentAgent.\n"
    "5. Based on the alignment details, decide the next step and formulate a response or action.\n"
    "    - If out_of_scope, inform the user politely.\n"
    "    - If relevant to a different subject, clarify with the user.\n"
    "    - If aligned, determine intent (e.g., conceptual help, practice, diagnosis) and route to a specialist agent.\n"
    "Communication Guidelines with CurriculumAlignmentAgent:\n"
    "- Your message TO the CurriculumAlignmentAgent MUST be a JSON string.\n"
    "- You will receive a JSON string as a response FROM the CurriculumAlignmentAgent.\n"
    "- Your reply in the main chat will be your analysis and intended next steps, directly addressing the UserProxy."
    "\n\nKey new responsibilities:\n"
    "- After asking the user to clarify subject context, await their response (SIMULATED FOR NOW).\n"
    "- If they confirm a switch, acknowledge and (conceptually) prepare to route to a specialist (e.g., ConceptTutor) with the new context.\n"
    "- If they decline or ask a new question, re-initiate curriculum alignment for the new query.\n"
    "- When concluding your turn based on current logic (e.g., after alignment check), end your message with \"exit\" to signal termination to the UserProxy in test mode."
    "\n\nKey responsibilities updated:\n"
    "- If query is aligned with current context: Route using the query and alignment details to the ConceptTutorAgent.\n"
    "- If query seems like a request for practice: Route to AssessmentRevisionAgent.\n"
    "- If query seems like a request to check an answer: Route to DiagnosticRemediationAgent.\n"
    "- If query relates to CALA projects: Route to ProjectsMentorAgent.\n"
    "- If query asks to generate content (notes, diagrams): Route to ContentGenerationAgent.\n"
    "- If query asks about progress/scores: Route to AnalyticsProgressAgent.\n"
    "- Your final reply to the UserProxy should be the response received from the specialist agent."
)

        # code_execution_config will be passed via kwargs from main.py
        super().__init__(name,
                         system_message=system_message,
                         llm_config=llm_config,
                         **kwargs)
        self.curriculum_alignment_agent = curriculum_alignment_agent
        self.concept_tutor_agent = concept_tutor_agent
        self.diagnostic_agent = diagnostic_agent
        self.assessment_agent = assessment_agent
        self.projects_mentor_agent = projects_mentor_agent
        self.content_generation_agent = content_generation_agent
        self.analytics_progress_agent = analytics_progress_agent
        
        self.current_session_alignment = None # This will now store a dict like {'data': alignment_data, 'timestamp': float, 'last_system_response': str, 'last_user_query': str}
        self._active_sub_chat_partner_name = None # Initialize new instance variable
        self.stale_threshold_seconds = STALE_THRESHOLD_SECONDS # Configurable staleness threshold
        
        # Register the main reply generation function
        self.register_reply(
            autogen.Agent, 
            OrchestratorAgent._generate_orchestrator_reply
        )
        
        # Removed registration of project_checklist tool from Orchestrator

        self.stage_timings = {}
        self.token_counts = {'input': 0, 'output': 0, 'total': 0}
        self.first_token_time = None
        # self._intent_classifier_client is no longer needed as we use self.client

    async def _classify_intent_with_llm(self, current_query: str, last_system_response: Optional[str], last_user_query: Optional[str], full_history: Optional[List[Dict[str, str]]], alignment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Uses the agent's configured LLM client to classify the intent of the current user query.

        This method constructs a detailed prompt for the LLM, including chat history
        and current syllabus context (if available from alignment_data), to determine
        if the user's latest query is an affirmative/negative follow-up, a request for
        clarification on the current topic, a new unrelated topic, or ambiguous but contextual.

        Args:
            current_query (str): The user's most recent query.
            last_system_response (Optional[str]): The last response sent by the system.
            last_user_query (Optional[str]): The user's query preceding the current one.
            full_history (Optional[List[Dict[str, str]]]): The entire conversation history.
            alignment_data (Optional[Dict[str, Any]]): Current syllabus alignment data, which may
                                                      contain `raw_metadata_preview` with `page_content`
                                                      for syllabus context.

        Returns:
            Dict[str, Any]: A dictionary containing the classified intent, a confidence score,
                            and the raw response from the LLM. Example:
                            {'intent': 'AFFIRMATIVE_FOLLOW_UP', 'confidence': 0.8, 'raw_response': 'AFFIRMATIVE_FOLLOW_UP'}
        """
        # Check if the agent's client is available. 
        # self.client is initialized by ConversableAgent based on llm_config.
        if not hasattr(self, 'client') or self.client is None:
            logger.warning("Agent's LLM client (self.client) not available for intent classification.")
            return {"intent": "UNKNOWN", "confidence": 0.0, "raw_response": "Agent client missing"}

        # Format chat history with timestamps
        chat_log = []
        if full_history:
            for msg in full_history:
                role = msg.get("role", "user") if msg.get("name") != self.name else "assistant"
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", datetime.utcnow().isoformat() + "Z")
                
                if isinstance(content, str):
                    try:
                        parsed_c = json.loads(content)
                        if isinstance(parsed_c, dict) and "user_query" in parsed_c:
                            content = parsed_c.get("original_user_query", parsed_c.get("user_query"))
                        elif isinstance(parsed_c, dict) and "answer" in parsed_c:
                            content = parsed_c.get("answer")
                    except json.JSONDecodeError:
                        pass
                
                chat_log.append({
                    "name": "Student" if role == "user" else "System",
                    "timestamp": timestamp,
                    "message": str(content)[:150]
                })

        # Add current query to chat log
        chat_log.append({
            "name": "Student",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "message": current_query
        })

        # Get syllabus_conetxt from alignment data if available
        syllabus_context = ""
        if alignment_data: # Check if alignment_data (the direct data dictionary) is available
            # Directly access raw_metadata_preview from the alignment_data argument
            raw_metadata = alignment_data.get("raw_metadata_preview", {})
            syllabus_context = raw_metadata.get("page_content", "")

        conversation_context = {
            "chat_log": chat_log,
            "syllabus_context": syllabus_context
        }

        system_prompt_content = (
            "You are an expert in classifying user intent in a tutoring conversation. "
            "Based on the provided chat log and syllabus_context, "
            "classify the most recent user message into one of these categories: "
            "1. AFFIRMATIVE_FOLLOW_UP: User is saying yes, okay, sure, etc., directly to a question or suggestion from the system. "
            "2. NEGATIVE_FOLLOW_UP: User is saying no, nope, etc., directly to a question or suggestion from the system. "
            "3. CLARIFICATION_ON_TOPIC: User is asking for more details, examples, or explanation about the *immediately preceding* topic, which might be informed by the provided chat_log or syllabus_context. "
            "4. NEW_TOPIC_UNRELATED: User is asking a question that seems unrelated to the immediate prior discussion AND is not covered by the provided syllabus_context. This indicates a potential need to fetch new information. "
            "5. AMBIGUOUS_CONTEXTUAL: User query is short (e.g. 'why', 'how') and context is needed to understand if it relates to prior topic or is new. Assume CLARIFICATION if contextually relevant. "
            "Respond with ONLY the classification label and nothing else. If unsure, use UNKNOWN."
        )
        
        prompt_message_objects = [
            SystemMessage(content=system_prompt_content),
            UserMessage(content=json.dumps(conversation_context), source="user")
        ]

        logger.debug(f"LLM Intent Classification Prompt Context:\n{json.dumps(conversation_context, indent=2)}")

        # Convert Pydantic message objects to a list of dicts for the API
        formatted_messages_for_api = []
        for msg_obj in prompt_message_objects:
            role = "system" if isinstance(msg_obj, SystemMessage) else "user" # Determine role
            formatted_messages_for_api.append({"role": role, "content": msg_obj.content})

        try:
            # Use the agent's existing client (self.client)
            # Wrap the client call in asyncio.to_thread to handle cases where self.client.create might be synchronous
            response = await asyncio.to_thread(
                self.client.create,
                messages=formatted_messages_for_api, # Pass the list of dicts
                cache=getattr(self, 'client_cache', None), # Use agent's cache if available
                temperature=0.1, 
                max_tokens=20 
            )
            
            raw_intent_label = ""
            # ModelClientResponseProtocol often has choices[0].message.content
            # or a direct text field depending on the client implementation used by ModelClient.
            if response.choices and len(response.choices) > 0 and response.choices[0].message and response.choices[0].message.content:
                raw_intent_label = response.choices[0].message.content.strip().upper()
            elif hasattr(response, 'text') and isinstance(response.text, str): # Fallback for some client responses
                 raw_intent_label = response.text.strip().upper()
            else:
                logger.warning(f"Could not extract content reliably from LLM response for intent. Response object: {response}")
                # Attempt to get any string part of the response if primary paths fail
                content_list = getattr(response, 'content', None)
                if isinstance(content_list, list) and content_list and isinstance(content_list[0], str):
                    raw_intent_label = content_list[0].strip().upper()
                elif isinstance(response.choices[0].message.tool_calls, list): # Handle cases where it tries to make a tool call like the main agent
                    logger.warning(f"LLM intent classifier attempted a tool call: {response.choices[0].message.tool_calls}")
                    raw_intent_label = "UNKNOWN_TOOL_CALL_ATTEMPT"

            logger.info(f"LLM Intent Classification Raw Label: {raw_intent_label}")

            confidence = 0.8 if raw_intent_label in ["AFFIRMATIVE_FOLLOW_UP", "NEGATIVE_FOLLOW_UP", "CLARIFICATION_ON_TOPIC", "NEW_TOPIC_UNRELATED", "AMBIGUOUS_CONTEXTUAL"] else 0.3
            
            if raw_intent_label == "AMBIGUOUS_CONTEXTUAL":
                logger.info("LLM classified as AMBIGUOUS_CONTEXTUAL, treating as CLARIFICATION_ON_TOPIC for reuse check.")

            return {"intent": raw_intent_label, "confidence": confidence, "raw_response": raw_intent_label if raw_intent_label else str(response)}
        except Exception as e:
            logger.error(f"Error during LLM intent classification: {e}", exc_info=True)
            return {"intent": "UNKNOWN", "confidence": 0.0, "raw_response": str(e)}

    async def _generate_orchestrator_reply(self, messages, sender, config):
        """
        Core reply generation logic for the OrchestratorAgent.

        This asynchronous method is triggered when the OrchestratorAgent receives a message.
        It orchestrates the entire response generation process:

        1.  **Initial Checks & Context Management**:
            - Ignores messages from active sub-chat partners to allow sub-chats to complete.
            - Prevents self-reply loops.
            - Parses the incoming message (expected JSON from `StudentInterfaceAgent`)
              to extract `original_user_query`, `form_level`, `context_was_reset`,
              and `previous_alignment_details`.
            - If `context_was_reset` is true or no `previous_alignment_details` are provided,
              clears `self.current_session_alignment`.

        2.  **Intent Classification & Context Reuse Decision**:
            - Retrieves `last_system_response` and `last_user_query` from `self.current_session_alignment`.
            - Calls `_classify_intent_with_llm` to determine the user's intent based on the
              current query, recent interactions, and full chat history.
            - Checks for simple conversational follow-ups (e.g., "yes", "thanks") using `is_short_conversational_follow_up`.
            - Checks if the system explicitly invited a follow-up using `did_system_invite_follow_up`.
            - Checks if the current session alignment is stale using `is_conversation_stale`.
            - Decides whether to reuse `self.current_session_alignment` based on intent, staleness,
              and whether the system invited a follow-up. Non-clarification intents or stale context
              typically lead to not reusing alignment.

        3.  **Syllabus Alignment (if needed)**:
            - If alignment is not being reused, initiates a chat with `CurriculumAlignmentAgent`,
              sending a JSON payload with `user_query` and `learner_context` (including `form_level`).
            - Parses the JSON response from `CurriculumAlignmentAgent`.
            - Updates `self.current_session_alignment` with the new alignment data, timestamp,
              current user query, and a placeholder for the system response.
            - Handles errors from the alignment agent (e.g., out of scope, JSON errors).

        4.  **Routing to Specialist Agent or Direct Reply**:
            - If the alignment indicates the query is out of scope, prepares a polite message for the user.
            - If the alignment suggests a different subject than the current context, it prepares
              a clarification question for the user (simulating the user's reply for now).
            - If the query is aligned and within scope:
                - Determines the most appropriate specialist agent based on keywords in the
                  user query (e.g., "explain" -> ConceptTutor, "practice" -> AssessmentRevision,
                  "project" -> ProjectsMentor, "diagram" -> ContentGeneration, etc.).
                - If no specific intent is matched, defaults to `ConceptTutorAgent`.
                - Initiates a chat with the chosen specialist agent, sending a payload containing
                  `original_user_query` and the `alignment_data`.
                - Parses the JSON response from the specialist agent, extracting the `answer`,
                  `retrieved_rag_context`, and `suggested_image_path`.

        5.  **Response Finalization & Payload Construction**:
            - Constructs the `final_orchestrator_output_payload` containing:
                - `answer`: The textual response for the user.
                - `retrieved_rag_context_for_answer`: RAG context used by the specialist.
                - `suggested_image_path`: Path to a suggested image, if any.
                - `alignment_data_used_for_routing`: The alignment data that informed the routing.
                - `orchestrator_dialogue_acts`: A log of decisions made by the orchestrator.
            - Updates `self.current_session_alignment['last_system_response']` with the final answer.
            - Logs timing and token count information.
            - Returns the JSON string of `final_orchestrator_output_payload`.

        Args:
            messages (List[Dict]): The list of messages. The last message is from `StudentInterfaceAgent`.
            sender (autogen.Agent): The agent that sent the message.
            config (Any): Optional configuration data.

        Returns:
            Tuple[bool, Union[str, None]]: (True, JSON string of the final payload) or (True, None) if no reply.
        """
        start_time = time.time()
        self.stage_timings = {}
        self.token_counts = {'input': 0, 'output': 0, 'total': 0}
        self.first_token_time = None

        # If this message is from an agent we're actively in a sub-chat with, ignore it here.
        if self._active_sub_chat_partner_name and self._active_sub_chat_partner_name == sender.name:
            logger.debug(f"Orchestrator: Received intermediate message from {sender.name} (active sub-chat partner). Letting sub-chat flow complete.")
            return True, None

        # Prevent Orchestrator from replying to its own messages
        if sender.name == self.name:
            logger.debug(f"Orchestrator: Message from self ({sender.name}), content: '{messages[-1].get('content', '')[:500]}...'. Declining to reply to prevent loop.")
            return True, None

        # Get the full history for potential LLM intent classification
        full_message_history_for_llm_intent = messages 
        last_message = messages[-1]
        user_input_raw = last_message.get("content", "")
        input_tokens = estimate_tokens(user_input_raw)
        self.token_counts['input'] += input_tokens
        self.token_counts['total'] += input_tokens
        
        logger.debug(f"\nOrchestrator: Received input content: '{user_input_raw}' from {sender.name}")
        logger.debug(f"Estimated input tokens: {input_tokens}")

        # Initialize variables for parsing and context
        user_query_for_alignment = user_input_raw
        user_query_for_specialist = user_input_raw
        form_level = "Form 4"
        current_subject = "Mathematics"
        original_user_query_from_payload = user_input_raw # To store the actual query if payload wraps it

        _previous_alignment_details_from_payload = None
        _context_was_reset_from_payload = False
        _last_system_response_from_payload = None # For context
        _last_user_query_from_payload = None # For context

        # Try to parse as JSON (standard for inter-agent comms in this system)
        try:
            payload = json.loads(user_input_raw)
            if isinstance(payload, dict):
                original_user_query_from_payload = payload.get("original_user_query", user_input_raw)
                user_query_for_specialist = payload.get("user_query", original_user_query_from_payload) 
                form_level = payload.get("form_level", form_level) 
                _previous_alignment_details_from_payload = payload.get("previous_alignment_details")
                _context_was_reset_from_payload = payload.get("context_was_reset", False) # Check for reset flag
                _last_system_response_from_payload = payload.get("last_system_response")
                _last_user_query_from_payload = payload.get("last_user_query")
                
                current_question_marker = "Student's Current Question:"
                if current_question_marker in user_query_for_specialist:
                    parts = user_query_for_specialist.split(current_question_marker)
                    if len(parts) > 1:
                        user_query_for_alignment = parts[1].strip()
                        logger.debug(f"Extracted current question for alignment: '{user_query_for_alignment}'")
                    else:
                        user_query_for_alignment = user_query_for_specialist 
                        logger.debug("Marker found, but split failed. Using full query for alignment as fallback.")
                else:
                    user_query_for_alignment = user_query_for_specialist
                
                logger.debug(f"Parsed JSON payload. Query for alignment: '{user_query_for_alignment}', Query for specialist: '{user_query_for_specialist[:100]}...', Form: '{form_level}', ContextReset: {_context_was_reset_from_payload}, HasPrevAlignInPayload: {bool(_previous_alignment_details_from_payload)}")
            else:
                logger.debug(f"Message was JSON, but not a dict. Using raw content as query: '{user_input_raw}'")
        except json.JSONDecodeError:
            logger.debug(f"Message not JSON. Using raw content as query: '{user_input_raw}'")

        # Define learner_context here so it's always available
        learner_context = {
            "current_subject_hint": current_subject,
            "current_form_level_hint": form_level
        }

        if _context_was_reset_from_payload:
            logger.info("Context_was_reset flag is true. Clearing Orchestrator's current_session_alignment.")
            self.current_session_alignment = None
        
        effective_previous_alignment_data = None
        last_interaction_timestamp = None
        previous_system_response_for_heuristic = _last_system_response_from_payload
        previous_user_query_for_heuristic = _last_user_query_from_payload

        if _previous_alignment_details_from_payload:
            logger.info("Using previous_alignment_details from incoming payload.")
            # Assuming payload structure for previous_alignment_details is now {'data': ..., 'timestamp': ..., 'last_system_response': ...}
            if isinstance(_previous_alignment_details_from_payload, dict):
                effective_previous_alignment_data = _previous_alignment_details_from_payload.get('data')
                last_interaction_timestamp = _previous_alignment_details_from_payload.get('timestamp')
                # Use system response from payload if available, otherwise it might be None
                previous_system_response_for_heuristic = _previous_alignment_details_from_payload.get('last_system_response', _last_system_response_from_payload)
                previous_user_query_for_heuristic = _previous_alignment_details_from_payload.get('last_user_query', _last_user_query_from_payload)

                # Sync agent state with the payload's full structure
                self.current_session_alignment = _previous_alignment_details_from_payload 
            else: # Old format, just the data
                effective_previous_alignment_data = _previous_alignment_details_from_payload
                # last_interaction_timestamp remains None, will be treated as stale or new
                self.current_session_alignment = {'data': effective_previous_alignment_data, 'timestamp': time.time(), 'last_system_response': None, 'last_user_query': None} # Upgrade old format
                logger.warning("Upgraded old format of previous_alignment_details. Timestamp set to now.")

        elif self.current_session_alignment and isinstance(self.current_session_alignment, dict):
            logger.info("Using Orchestrator's stored current_session_alignment as no previous_alignment_details in payload.")
            effective_previous_alignment_data = self.current_session_alignment.get('data')
            last_interaction_timestamp = self.current_session_alignment.get('timestamp')
            previous_system_response_for_heuristic = self.current_session_alignment.get('last_system_response', _last_system_response_from_payload)
            previous_user_query_for_heuristic = self.current_session_alignment.get('last_user_query', _last_user_query_from_payload)

        else:
            logger.info("No previous alignment details available (neither in payload nor in agent state).")

        # The system message now provides stronger guidance on reusing previous_alignment_details.
        # We will rely on the LLM's interpretation of the system message and the context (presence of previous_alignment_details).
        
        normalized_current_query = user_query_for_alignment.lower().strip().rstrip(".?!")

        alignment_data = None # This will store the actual alignment dictionary (the 'data' part)
        called_curriculum_alignment_this_turn = False
        should_call_curriculum_alignment = True # Default to calling
        reason_for_reuse_or_new_call = "Default: New query"
        llm_forced_new_alignment = False # Explicitly set for this path

        # --- Layered Follow-up Logic ---
        is_stale = is_conversation_stale(last_interaction_timestamp, self.stale_threshold_seconds)

        if is_stale:
            logger.info(f"Conversation is stale (last interaction timestamp: {last_interaction_timestamp}). Leaning towards new alignment or LLM check.")
            # If stale, we might bypass simple heuristics or require LLM confirmation even for short follow-ups.
            # For now, staleness primarily means we won't trust simple "yes" as much without more checks.
        
        # Layer 1: Heuristic Check (for non-stale, clear follow-ups)
        if effective_previous_alignment_data and not is_stale:
            is_short_follow_up = is_short_conversational_follow_up(normalized_current_query)
            system_invited = did_system_invite_follow_up(previous_system_response_for_heuristic)
            
            if is_short_follow_up and system_invited:
                logger.info(f"Heuristic Layer 1: Short follow-up ('{normalized_current_query}') to system invitation. Reusing previous alignment data.")
                alignment_data = effective_previous_alignment_data # Use the 'data' part
                should_call_curriculum_alignment = False
                called_curriculum_alignment_this_turn = False
                reason_for_reuse_or_new_call = "Heuristic: Short follow-up to system invite"
                # Update timestamp as context is actively used
                if self.current_session_alignment and isinstance(self.current_session_alignment, dict):
                    self.current_session_alignment['timestamp'] = time.time()
                    self.current_session_alignment['last_user_query'] = user_query_for_alignment # Update with the current "yes" etc.
                    # System response will be updated after specialist agent.
                
        # Layer 2: LLM Intent Classification (Placeholder) - if Layer 1 didn't apply or if stale and requires confirmation
        if should_call_curriculum_alignment: # Only if heuristic didn't catch it
            llm_intent_needed = False
            if is_stale and is_short_conversational_follow_up(normalized_current_query):
                logger.info("Query is a short follow-up but context is stale. LLM intent check is needed.")
                llm_intent_needed = True
            elif not is_short_conversational_follow_up(normalized_current_query): # Also consider for non-trivial queries not caught by simple heuristics
                # Potentially always call LLM if heuristic 1 fails, to be more robust, or add other conditions.
                # For now, let's trigger it if heuristic 1 failed and it's not a very short query (those might go to topic overlap).
                logger.info("Heuristic 1 failed for non-trivial query. Considering LLM intent check.")
                llm_intent_needed = True # Let's try LLM for more cases.
            
            if llm_intent_needed:
                logger.info("Calling LLM for intent classification...")
                intent_result = await self._classify_intent_with_llm(
                    user_query_for_alignment, 
                    previous_system_response_for_heuristic, 
                    previous_user_query_for_heuristic,
                    full_message_history_for_llm_intent, # Pass more history
                    effective_previous_alignment_data # Pass effective_previous_alignment_data for context
                ) 
                
                classified_intent = intent_result.get('intent')
                intent_confidence = intent_result.get('confidence', 0)
                reason_for_reuse_or_new_call = f"LLM Intent: {classified_intent} (Conf: {intent_confidence:.2f})"

                if classified_intent in ["AFFIRMATIVE_FOLLOW_UP", "NEGATIVE_FOLLOW_UP", "CLARIFICATION_ON_TOPIC", "AMBIGUOUS_CONTEXTUAL"] and intent_confidence > 0.6:
                    if effective_previous_alignment_data:
                        logger.info(f"LLM Intent: Classified as '{classified_intent}'. Reusing previous alignment data.")
                        alignment_data = effective_previous_alignment_data
                        should_call_curriculum_alignment = False
                        called_curriculum_alignment_this_turn = False
                        # Update timestamp as context is actively used by LLM decision
                        if self.current_session_alignment and isinstance(self.current_session_alignment, dict):
                            self.current_session_alignment['timestamp'] = time.time()
                            self.current_session_alignment['last_user_query'] = user_query_for_alignment 
                    else:
                        logger.warning("LLM suggested reuse based on intent, but no effective_previous_alignment_data found. Proceeding to new alignment.")
                        reason_for_reuse_or_new_call += "; No prev_alignment for reuse."
                        should_call_curriculum_alignment = True # Ensure it flips back if no data to reuse
                        llm_forced_new_alignment = False # Explicitly set for this path
                elif classified_intent == "NEW_TOPIC_UNRELATED" and intent_confidence > 0.6:
                    logger.info(f"LLM Intent: Classified as NEW_TOPIC_UNRELATED. Proceeding to new alignment.")
                    should_call_curriculum_alignment = True
                    llm_forced_new_alignment = True # LLM forces new alignment
                else:
                    logger.info(f"LLM Intent: Classified as '{classified_intent}' or low confidence. Fallback to topic overlap / new alignment.")
                    # should_call_curriculum_alignment remains true or as per previous logic if LLM is unsure.
                    llm_forced_new_alignment = False # LLM did not force new alignment
            

        # Fallback/Existing Logic: Topic Overlap Check (if still needing to decide on new call)
        if should_call_curriculum_alignment and effective_previous_alignment_data and not llm_forced_new_alignment:
            # Topic overlap check: compare new query words against page_content + chat history
            topic_page_content = effective_previous_alignment_data.get("raw_metadata_preview", {}).get("page_content", "")
            # Use user_query_for_specialist for history as it contains the fuller context from payload
            history_text = user_query_for_specialist.lower() if isinstance(user_query_for_specialist, str) else ""
            
            topic_and_history = topic_page_content.lower() + " " + history_text
            topic_words = set(re.findall(r"\w+", topic_and_history))
            query_words = set(re.findall(r"\w+", user_query_for_alignment.lower()))
            overlap = topic_words.intersection(query_words)
            min_overlap = 3  # require at least 3 overlapping words for reuse (or make this configurable)
            
            # Be more conservative if the query is very short and not caught by Layer 1
            is_very_short_query = len(user_query_for_alignment.split()) <= 2

            if len(overlap) >= min_overlap and not is_very_short_query : # Avoid reusing for very short queries like "yes" if they slipped through Layer 1
                logger.info(f"Topic overlap check passed ({len(overlap)} overlapping words). Reusing previous alignment data.")
                alignment_data = effective_previous_alignment_data
                called_curriculum_alignment_this_turn = False
                should_call_curriculum_alignment = False
                reason_for_reuse_or_new_call = f"Heuristic: Topic overlap ({len(overlap)} words)"
                if self.current_session_alignment and isinstance(self.current_session_alignment, dict): # Update timestamp
                     self.current_session_alignment['timestamp'] = time.time()
                     self.current_session_alignment['last_user_query'] = user_query_for_alignment
            elif is_very_short_query and len(overlap) >= min_overlap:
                 logger.info(f"Topic overlap passed ({len(overlap)} words) but query ('{user_query_for_alignment}') is very short. Preferring new alignment or LLM check if stale.")
                 reason_for_reuse_or_new_call = "Heuristic: Topic overlap but query too short, considering new alignment."
                 # should_call_curriculum_alignment remains true


        if should_call_curriculum_alignment:
            if effective_previous_alignment_data:
                 logger.info(f"Reason for new alignment: {reason_for_reuse_or_new_call}. Original query for alignment: '{normalized_current_query}'")
            else: 
                 logger.info(f"No effective previous alignment. Reason for new alignment: New session/context. Original query: '{normalized_current_query}'")

            logger.debug(f"Calling CurriculumAlignmentAgent for: '{user_query_for_alignment}'")
            message_for_alignment_agent = json.dumps({
                "user_query": user_query_for_alignment, # Use the extracted current question
                "learner_context": learner_context
            })
            
            logger.debug(f"Sending to CurriculumAlignmentAgent: {message_for_alignment_agent}")

            alignment_start_time = time.time()
            chat_results_alignment = await self.a_initiate_chat(
                recipient=self.curriculum_alignment_agent,
                message=message_for_alignment_agent,
                max_turns=1, 
                summary_method="last_msg",
                silent=False 
            )
            alignment_processing_time = time.time() - alignment_start_time
            self.stage_timings['curriculum_alignment'] = alignment_processing_time
            called_curriculum_alignment_this_turn = True

            alignment_json_str = chat_results_alignment.summary
            alignment_tokens_count = estimate_tokens(alignment_json_str if alignment_json_str else "")
            self.token_counts['input'] += alignment_tokens_count # Considered as input for Orchestrator's processing
            self.token_counts['total'] += alignment_tokens_count

            logger.debug(f"Curriculum Alignment Stage (called: {called_curriculum_alignment_this_turn}):")
            logger.debug(f"- Time taken: {alignment_processing_time:.2f} seconds")
            logger.debug(f"- Estimated tokens: {alignment_tokens_count}")

            try:
                alignment_data = json.loads(alignment_json_str)
                logger.debug(f"Parsed new alignment data: {json.dumps(alignment_data, indent=2)}")
                # Update current_session_alignment with new data and timestamp
                self.current_session_alignment = {
                    'data': alignment_data, 
                    'timestamp': time.time(),
                    'last_system_response': None, # Will be set after specialist replies
                    'last_user_query': user_query_for_alignment
                }
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error decoding JSON from CurriculumAlignmentAgent: {e}. Received: {alignment_json_str}")
        
        # Ensure alignment_data is not None before proceeding AND before storing it in agent state
        if alignment_data: # Check if alignment_data is not None (could be from reuse or new call)
            if not alignment_data.get("error"): # Only store if it's not an error alignment
                # current_session_alignment is updated either during reuse or after successful new call
                pass # logger.debug(f"Orchestrator's current_session_alignment is up-to-date.")
            else:
                logger.warning(f"Current turn's alignment_data indicates an error: {alignment_data.get('error')}. Not updating current_session_alignment with this error state, or it was already an error from reuse.")
        else: # alignment_data is None at this point - this should be rare now.
             logger.warning("alignment_data is None for this turn (either new call failed/skipped or reuse was not applicable).")

        reply_to_user = ""
        terminate_signal = " exit" 
        next_agent = None 
        message_for_next_agent = ""
        final_system_response_for_session_log = None # To store the reply that will become 'last_system_response'

        if alignment_data is None: # This should be handled by fallbacks, but as a safety.
            reply_to_user = "I couldn't get syllabus alignment information. Please try again."
        elif alignment_data.get("error") == "out_of_scope" or not alignment_data.get("is_in_syllabus"):
            reply_to_user = "It seems your question might be outside the scope of the ZIMSEC O-Level syllabus I cover."
            if alignment_data.get("notes_for_orchestrator"):
                 reply_to_user += f" ({alignment_data.get('notes_for_orchestrator')})"
            reply_to_user += " Do you have a question related to ZIMSEC Mathematics or Combined Science?"
            # For out_of_scope, no RAG context is typically generated by a specialist, so pass None
            retrieved_rag_context_for_final_reply = alignment_data.get("raw_metadata_preview") # or None if not relevant

        elif alignment_data.get("identified_subject") and alignment_data.get("identified_subject") != learner_context.get("current_subject_hint"):
            identified_subject = alignment_data.get('identified_subject')
            topic_info = f"{alignment_data.get('identified_topic', 'N/A')} / {alignment_data.get('identified_subtopic', 'N/A')}"
            question_to_user = (
                f"That's an interesting question! It looks like it aligns best with **{identified_subject}** under the topic: *{topic_info}*. "
                f"Our current context is {learner_context.get('current_subject_hint')}. "
                f"Would you like to switch our focus to {identified_subject} to discuss this? (yes/no)"
            )
            logger.debug(f"\nOrchestrator would ask user: {question_to_user}")
            logger.debug("\nOrchestrator: SIMULATING user replied 'yes' to context switch.")
            simulated_user_reply = "yes"
            if simulated_user_reply.lower() in ["yes", "ok", "sure", "yep", "please do", "switch"]:
                # User wants to switch. Update learner context.
                learner_context["current_subject_hint"] = identified_subject
                # For now, we'll just acknowledge. A more advanced flow might re-route or re-align.
                # The key is, the specialist will now receive the original user_query_for_specialist which might contain history,
                # and the alignment_data now reflects the NEWLY confirmed topic/subject for RAG.
                reply_to_user = f"Great! Switching context to **{identified_subject}**... Let's discuss your question about '{user_query_for_alignment[:30]}...'."
                # Potentially, we could now directly route to a specialist with user_query_for_specialist and the new alignment_data.
                # For this example, we might just set up for the next turn or route to ConceptTutor with the updated context.
                next_agent = self.concept_tutor_agent
                message_for_next_agent = json.dumps({
                    "original_user_query": user_query_for_specialist, # Full query with history
                    "alignment_data": alignment_data, # Alignment for the NEW subject/topic
                    "previous_system_response": previous_system_response_for_heuristic, # Pass context
                    "user_follow_up_query": user_query_for_alignment, # The query that triggered this path
                    "updated_orchestrator_session_alignment_for_next_turn": self.current_session_alignment # Full context for NEXT turn
                })
                logger.debug(f"Context switched. Routing to ConceptTutorAgent with new alignment for: '{user_query_for_specialist[:100]}...'")
                final_system_response_for_session_log = reply_to_user # Store for next turn's context

            else: # User declined to switch context
                 reply_to_user = "Okay, we'll stick to the current context. Do you have another question related to {current_subject}?"
            retrieved_rag_context_for_final_reply = alignment_data.get("raw_metadata_preview") 
        else:
            subject = alignment_data.get('identified_subject', 'the syllabus')
            topic_info = f"{alignment_data.get('identified_topic', 'N/A')} / {alignment_data.get('identified_subtopic', 'N/A')}"
            logger.debug(f"Query aligned with {subject} - Topic: {topic_info}. Determining intent...")

            # Initialize retrieved_rag_context_for_final_reply, default to alignment RAG if no specialist is called
            retrieved_rag_context_for_final_reply = alignment_data.get("raw_metadata_preview")

            query_lower = user_query_for_alignment.lower()
            if any(kw in query_lower for kw in ["practice", "quiz"]):
                next_agent = self.assessment_agent
                message_for_next_agent = json.dumps({
                    "task": "generate_questions", "topic": alignment_data.get('identified_topic', 'mixed'),
                    "subtopic": alignment_data.get('identified_subtopic'), "form": form_level,
                    "difficulty": "medium", "n": 3
                })
                logger.debug(f"Intent detected: Practice/Assess. Routing to {next_agent.name}.")
            elif any(kw in query_lower for kw in ["check my answer", "grade my work", "diagnose this", "diagnose my understanding"]):
                next_agent = self.diagnostic_agent
                # Get diagnostic check from initial message if present
                diagnostic_check = {}
                try:
                    initial_data = json.loads(user_input_raw)
                    diagnostic_check = initial_data.get("diagnostic_check", {})
                except json.JSONDecodeError:
                    pass
                
                message_for_next_agent = json.dumps({
                    "task": "diagnose_answer",
                    "question": f"Regarding {topic_info}",
                    "learner_answer": user_query_for_alignment,
                    "marking_rubric": {},
                    "diagnostic_check": diagnostic_check,
                    "alignment_data": alignment_data
                })
                logger.debug(f"Intent detected: Diagnose/Check Answer. Routing to {next_agent.name}.")
            elif any(kw in query_lower for kw in ["project", "cala", "milestone", "research"]):
                next_agent = self.projects_mentor_agent
                message_for_next_agent = json.dumps({
                    "task": "project_guidance", "milestone": "plan", 
                    "draft_snippet": user_query_for_alignment, # Use current question for project task
                    "full_query_context": user_query_for_specialist # Provide full context
                })
                logger.debug(f"Intent detected: Project/CALA. Routing to {next_agent.name}.")
            elif any(kw in query_lower for kw in ["generate notes", "create worksheet", "make diagram"]):
                next_agent = self.content_generation_agent
                asset_type = "notes"; # Default, can be refined by LLM
                if "worksheet" in query_lower: asset_type = "worksheet"
                elif "diagram" in query_lower: asset_type = "diagram"
                message_for_next_agent = json.dumps({
                    "task": "generate_content", 
                    "asset_type": asset_type, 
                    "topic": alignment_data.get('identified_topic', 'general'),
                    "subtopic": alignment_data.get('identified_subtopic'), 
                    "form": form_level,
                    "user_query": user_query_for_alignment, # Base generation on the current question
                    "full_query_context": user_query_for_specialist # Provide full context
                })
                logger.debug(f"Intent detected: Content Generation. Routing to {next_agent.name}.")
            elif any(kw in query_lower for kw in ["progress", "scores", "analytics"]):
                next_agent = self.analytics_progress_agent
                message_for_next_agent = json.dumps({
                    "task": "report_progress", "student_id": "student_001" # Placeholder student ID
                })
                logger.debug(f"Intent detected: Analytics/Progress. Routing to {next_agent.name}.")
            else:
                # Default to ConceptTutorAgent if no other specific intent is matched
                next_agent = self.concept_tutor_agent
                message_for_next_agent = json.dumps({
                    "original_user_query": user_query_for_specialist, # Pass the full query (potentially augmented)
                    "alignment_data": alignment_data, # Based on the current question's alignment
                    "previous_system_response": previous_system_response_for_heuristic if not called_curriculum_alignment_this_turn else None,
                    "user_follow_up_query": user_query_for_alignment if not called_curriculum_alignment_this_turn else None
                })
                logger.debug(f"Defaulting to ConceptTutorAgent. Routing to {next_agent.name}.")
            
            # Initialize variable to hold potential image path from specialist
            suggested_image_path_from_specialist = None

            if next_agent:
                logger.debug(f"Orchestrator: Initiating chat with {next_agent.name} with message: {message_for_next_agent[:200]}...")
                self._active_sub_chat_partner_name = next_agent.name # Set active partner
                specialist_chat_start_time = time.time()
                chat_results_specialist = await self.a_initiate_chat(
                    recipient=next_agent,
                    message=message_for_next_agent,
                    max_turns=3, # Allow specialist agent to have a few turns if needed (e.g. tool use)
                    summary_method="last_msg", # We want the last message content
                    silent=False
                )
                specialist_chat_time = time.time() - specialist_chat_start_time
                self.stage_timings[f'specialist_chat_{next_agent.name}'] = specialist_chat_time
                self._active_sub_chat_partner_name = None # Clear active partner

                specialist_response_summary = chat_results_specialist.summary
                specialist_tokens = estimate_tokens(specialist_response_summary if specialist_response_summary else "")
                self.token_counts['input'] += specialist_tokens # Specialist response is input to Orchestrator's decision
                self.token_counts['total'] += specialist_tokens

                logger.debug(f"Specialist ({next_agent.name}) Chat Stage:")
                logger.debug(f"- Time taken: {specialist_chat_time:.2f} seconds")
                logger.debug(f"- Estimated tokens: {specialist_tokens}")
                logger.debug(f"- Specialist Raw Summary: {specialist_response_summary}")

                if specialist_response_summary:
                    try:
                        # Attempt to parse the specialist's response as our expected JSON payload
                        parsed_specialist_response = json.loads(specialist_response_summary)
                        if isinstance(parsed_specialist_response, dict):
                            reply_to_user = parsed_specialist_response.get("answer", specialist_response_summary)
                            retrieved_rag_context_for_final_reply = parsed_specialist_response.get("retrieved_rag_context")
                            suggested_image_path_from_specialist = parsed_specialist_response.get("suggested_image_path") # Extract image path
                            final_system_response_for_session_log = reply_to_user # Store for next turn's context
                            logger.debug("Successfully parsed structured response from specialist.")
                        else:
                            # Specialist returned JSON, but not the expected dict structure
                            reply_to_user = specialist_response_summary
                            final_system_response_for_session_log = reply_to_user
                            # RAG context from specialist is not in the expected format
                            logger.warning("Specialist returned JSON but not in expected answer/context format.")
                    except json.JSONDecodeError:
                        # Specialist returned a plain string, not JSON
                        reply_to_user = specialist_response_summary
                        final_system_response_for_session_log = reply_to_user
                        # RAG context from specialist is not available if it wasn't JSON
                        logger.debug("Specialist returned a plain string response.")
                else:
                    reply_to_user = f"I consulted with the {next_agent.name}, but didn't get a conclusive answer. Could you rephrase?"
                    final_system_response_for_session_log = reply_to_user
                    # No RAG context from specialist if no summary
                    logger.warning(f"No summary received from {next_agent.name}.")
            else:
                # This case should ideally not be reached if ConceptTutor is the default
                reply_to_user = "I've analyzed your query with the syllabus, but I'm not sure how to proceed. Can you clarify?"
                final_system_response_for_session_log = reply_to_user
                logger.warning("No specialist agent was selected after alignment.")

        self.first_token_time = time.time() # Rough approximation for TTFT from Orchestrator's final processing
        output_tokens = estimate_tokens(reply_to_user)
        self.token_counts['output'] += output_tokens
        self.token_counts['total'] += output_tokens

        logger.debug(f"\nOrchestrator Final Reply to UserProxy (via StudentInterfaceAgent): '{reply_to_user[:200]}...'")
        logger.debug(f"Orchestrator RAG context for final reply: '{str(retrieved_rag_context_for_final_reply)[:200]}...'")
        logger.debug(f"Estimated output tokens for orchestrator's reply: {output_tokens}")
        logger.debug(f"Total session tokens (Orchestrator perspective): Input={self.token_counts['input']}, Output={self.token_counts['output']}, Total={self.token_counts['total']}")
        total_processing_time = time.time() - start_time
        self.stage_timings['total_orchestrator_processing'] = total_processing_time
        logger.debug(f"Total Orchestrator processing time: {total_processing_time:.2f} seconds. Stage timings: {self.stage_timings}")
        
        # Update current_session_alignment with the final system response for context in the next turn
        if self.current_session_alignment and isinstance(self.current_session_alignment, dict) and final_system_response_for_session_log:
            self.current_session_alignment['last_system_response'] = final_system_response_for_session_log
            # 'last_user_query' and 'timestamp' should have been updated when alignment_data was set or reused.
        
        # Package the final response for main.py
        # Ensure alignment_data_to_use is defined. It would be the result from CurriculumAlignmentAgent or reused previous alignment.
        # For this change, we are ensuring that 'alignment_data' (which holds the active alignment info) is passed.
        
        # We pass self.current_session_alignment because it now holds the full context including timestamp and last responses
        # The recipient (StudentInterfaceAgent -> main.py) will need to know how to unpack this if it needs individual pieces,
        # or it can just pass the whole self.current_session_alignment as 'previous_alignment_details' in the next turn.
        # For routing purposes in THIS turn, 'alignment_data' (the 'data' part) was used.
        final_orchestrator_output_payload = {
            "answer": reply_to_user, # This is the specialist's answer, or Orchestrator's direct reply
            "retrieved_rag_context_for_answer": retrieved_rag_context_for_final_reply, # RAG context associated with 'answer'
            "suggested_image_path": suggested_image_path_from_specialist, # Add the image path here
            "alignment_data_used_for_routing": alignment_data, # The 'data' part of the alignment used for routing THIS turn
            "updated_orchestrator_session_alignment_for_next_turn": self.current_session_alignment # Full context for NEXT turn
        }
        return True, json.dumps(final_orchestrator_output_payload) + terminate_signal

if __name__ == '__main__':
    # This testing block is more complex now as Orchestrator initiates chats.
    # It's better to test this interaction within the main_script.py setup.
    # However, a simplified direct call might look like this if other agents are minimal.
    
    print("OrchestratorAgent self-test (conceptual - real test via main.py recommended)")
    
    # Minimal config for testing agent instantiation
    config_list_test = [
        {
            "model": "gpt-4.1-nano",
            "api_key": os.environ.get("OPENAI_API_KEY")
        }
    ]

    from curriculum_alignment_agent import CurriculumAlignmentAgent

    # Create a dummy UserProxyAgent that the Orchestrator can use to send messages *if needed by a_initiate_chat's internals*
    # although self.a_initiate_chat should use Orchestrator's own capabilities.
    # For this test, the main thing is that CurriculumAlignmentAgent needs to be able to reply.
    
    curriculum_agent_for_test = CurriculumAlignmentAgent(
        name="CurriculumAgentForOrchestratorTest",
        llm_config={"config_list": config_list_test} # It uses its own llm_config for its system message, but reply is rule-based
    )

    orchestrator = OrchestratorAgent(
        name="OrchestratorIsolatedTest",
        llm_config={"config_list": config_list_test},
        curriculum_alignment_agent=curriculum_agent_for_test,
        concept_tutor_agent=curriculum_agent_for_test,
        diagnostic_agent=curriculum_agent_for_test,
        assessment_agent=curriculum_agent_for_test,
        projects_mentor_agent=curriculum_agent_for_test,
        content_generation_agent=curriculum_agent_for_test,
        analytics_progress_agent=curriculum_agent_for_test
    )

    # To test _generate_orchestrator_reply, we need an event loop and to simulate an incoming message.
    async def test_orchestrator_reply():
        print("\n--- Testing Orchestrator with Physics Query ---")
        mock_user_message_physics = {"content": "What is the difference between speed and velocity?"}
        # The sender argument is important for context in more complex scenarios, here can be None or a dummy agent
        # We are calling the method directly, not going through full Autogen receive sequence
        # For a more integrated test, we'd use a UserProxyAgent to send to orchestrator.
        # This test assumes the orchestrator's reply function is correctly triggered.
        
        # Let's simulate a UserProxyAgent sending the message to the Orchestrator
        # This is closer to how it would work in a group chat or with initiate_chat
        user_proxy_sim = autogen.UserProxyAgent("SimUserProxy", human_input_mode="NEVER", code_execution_config=False)

        # The orchestrator will initiate a chat with the curriculum agent.
        # The curriculum agent replies with JSON (no LLM needed for its reply due to registered function).
        # The orchestrator itself might use an LLM for its system message interpretation if it was more complex,
        # but its current reply generation is also rule-based after getting curriculum info.

        # We'll test by having the user_proxy_sim initiate a 1-turn chat with orchestrator
        await user_proxy_sim.a_initiate_chat(
            recipient=orchestrator,
            message=mock_user_message_physics["content"],
            max_turns=1 # User sends, Orchestrator processes and replies
        )

    # asyncio.run(test_orchestrator_reply()) # This test setup needs refinement to properly capture reply.
    # The previous test in orchestrator_agent.py (using process_user_query directly) was better for isolated unit testing of that core logic.
    # The true test of this refactored orchestrator will be in main.py.
    print("Skipping direct test of refactored Orchestrator. Test via main.py.") 