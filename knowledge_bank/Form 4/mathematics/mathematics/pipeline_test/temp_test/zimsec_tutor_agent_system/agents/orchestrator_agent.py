# orchestrator_agent.py
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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Removed constants EXAMPLE_PROJECT_TOPICS, STAGE_1_GUIDELINES from here

# Helper function to estimate tokens (rough approximation)
def estimate_tokens(text: str) -> int:
    """Rough estimate of tokens based on characters and words."""
    # Average of 4 chars per token and 1.3 tokens per word
    char_estimate = len(text) / 4
    word_estimate = len(text.split()) * 1.3
    return int((char_estimate + word_estimate) / 2)

class OrchestratorAgent(autogen.AssistantAgent):
    def __init__(self, name, llm_config, curriculum_alignment_agent, concept_tutor_agent, diagnostic_agent, assessment_agent, projects_mentor_agent, content_generation_agent, analytics_progress_agent, **kwargs):
        system_message = (
    "You are the Orchestrator Agent in a multi-agent AI tutoring system for ZIMSEC O-Level students.\n"
    "Your primary roles are:\n"
    "1. Receive the initial user query from the UserProxy.\n"
    "2. ALWAYS first consult the CurriculumAlignmentAgent. You will send it a JSON message like: {\"user_query\": \"<actual_user_query>\", \"learner_context\": {\"current_subject_hint\": \"<current_subject_or_default>\"}}.\n"
    "3. Receive and parse the JSON response from the CurriculumAlignmentAgent.\n"
    "4. Based on the alignment details, decide the next step and formulate a response or action.\n"
    "    - If out_of_scope, inform the user politely.\n"
    "    - If relevant to a different subject, clarify with the user.\n"
    "    - If aligned, (for now) state this and that you would next determine intent and route to a specialist.\n"
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
        
        self.current_session_alignment = None # Added to store alignment within a session
        self._active_sub_chat_partner_name = None # Initialize new instance variable
        
        # Register the main reply generation function
        self.register_reply(
            autogen.Agent, 
            OrchestratorAgent._generate_orchestrator_reply
        )
        
        # Removed registration of project_checklist tool from Orchestrator

        self.stage_timings = {}
        self.token_counts = {
            'input': 0,
            'output': 0,
            'total': 0
        }
        self.first_token_time = None

    async def _generate_orchestrator_reply(self, messages, sender, config):
        """
        This function is called when the OrchestratorAgent receives a message.
        It processes the user query, consults curriculum alignment, and decides next steps.
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
            logger.debug(f"Orchestrator: Message from self ({sender.name}), content: '{messages[-1].get('content', '')[:100]}...'. Declining to reply to prevent loop.")
            return True, None

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

        _previous_alignment_details_from_payload = None
        _context_was_reset_from_payload = False

        # Try to parse as JSON (standard for inter-agent comms in this system)
        try:
            payload = json.loads(user_input_raw)
            if isinstance(payload, dict):
                user_query_for_specialist = payload.get("user_query", user_input_raw) 
                form_level = payload.get("form_level", form_level) 
                _previous_alignment_details_from_payload = payload.get("previous_alignment_details")
                _context_was_reset_from_payload = payload.get("context_was_reset", False) # Check for reset flag
                
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
        
        effective_previous_alignment_details = None
        if _previous_alignment_details_from_payload:
            logger.info("Using previous_alignment_details from incoming payload.")
            effective_previous_alignment_details = _previous_alignment_details_from_payload
            self.current_session_alignment = _previous_alignment_details_from_payload # Sync agent state
        elif self.current_session_alignment:
            logger.info("Using Orchestrator's stored current_session_alignment as no previous_alignment_details in payload.")
            effective_previous_alignment_details = self.current_session_alignment
        else:
            logger.info("No previous alignment details available (neither in payload nor in agent state).")


        # Ensure user_query_for_alignment is not empty after potential extraction
        if not user_query_for_alignment.strip():
            # This might happen if "Student's Current Question:" is present but empty.
            logger.warning("Extracted query for alignment is empty. Falling back to user_query_for_specialist.")
            user_query_for_alignment = user_query_for_specialist # Fallback to the full query to avoid sending empty to alignment.
            # It might be better to just respond with an error or ask for clarification if the current question is truly empty.
            # For now, this fallback prevents an error with the alignment agent.
            if not user_query_for_alignment.strip(): # Double check if specialist query is also empty
                 logger.debug("Received empty query (no content and no tool_calls). Ending turn.")
                 return True, "Looks like there was no question. Ask me something else! exit"


        tool_calls_from_message = last_message.get("tool_calls")
        # Check if the query intended for alignment is just whitespace AND there are no tool calls
        if not user_query_for_alignment.strip() and not tool_calls_from_message:
            logger.debug("Effective query for alignment is empty (no content and no tool_calls). Ending turn.")
            return True, "Looks like there was no question. Ask me something else! exit"

        # --- Logic for reusing previous alignment for generic follow-ups ---
        # GENERIC_FOLLOW_UP_PHRASES = [
        #     "teach me more", "explain further", "more details", "go on", "continue",
        #     "next section", "this section", "tell me more", "more about this",
        #     "elaborate", "expand on that", "what else", "anything more",
        #     "can you teach me more", "can you explain further", "tell me more about this",
        #     "what about this section", "more from this section", "teach me more from this section",
        #     "explain more from this section", "can you elaborate", "can you expand on that",
        #     "show me more", "give me more details", "let's continue", "proceed",
        #     "more on this topic", "more about that topic", "continue with this topic"
        # ]
        # The system message now provides stronger guidance on reusing previous_alignment_details.
        # We will rely on the LLM's interpretation of the system message and the context (presence of previous_alignment_details).
        
        normalized_current_query = user_query_for_alignment.lower().strip().rstrip(".?!")

        alignment_data = None
        called_curriculum_alignment_this_turn = False
        should_call_curriculum_alignment = True # Default to calling

        if effective_previous_alignment_details:
            # Heuristic 1: Very short query
            very_short_query_threshold = 4 # words (e.g., "tell me more", "go on please")
            if len(normalized_current_query.split()) <= very_short_query_threshold: # Changed to <=
                logger.info(f"VERY SHORT query ('{normalized_current_query}') detected with previous alignment. Reusing previous alignment details definitively.")
                alignment_data = effective_previous_alignment_details
                called_curriculum_alignment_this_turn = False
                should_call_curriculum_alignment = False
            
            # Heuristic 2 (if Heuristic 1 didn't trigger and should_call_curriculum_alignment is still true):
            # More nuanced continuation check. This replaces the previous SIMPLE_CONTINUATION_PHRASES logic.
            if should_call_curriculum_alignment: 
                CONTINUATION_KEYWORDS = [
                    "explain", "further", "more", "detail", "details", "continue", "next", 
                    "tell", "elaborate", "expand", "what else", "anything",
                    "this", "that", "it", 
                    "section", "topic", "part", "point", "aspect", "previous", "last"
                ]
                query_words_set = set(normalized_current_query.lower().split())
                
                matched_keywords = query_words_set.intersection(CONTINUATION_KEYWORDS)
                
                min_keyword_matches = 2 
                min_keyword_ratio = 0.35 
                max_total_words_for_heuristic = 12 

                num_query_words = len(query_words_set)

                if num_query_words > 0 and num_query_words <= max_total_words_for_heuristic and \
                   len(matched_keywords) >= min_keyword_matches and \
                   (len(matched_keywords) / num_query_words) >= min_keyword_ratio:
                    
                    logger.info(f"Continuation-like query ('{normalized_current_query}') detected by keyword heuristic. Matched keywords: {matched_keywords}. Query words: {num_query_words}. Reusing previous alignment details.")
                    alignment_data = effective_previous_alignment_details
                    called_curriculum_alignment_this_turn = False
                    should_call_curriculum_alignment = False
                # else:
                #    logger.info(f"Query ('{normalized_current_query}') did not meet keyword heuristic for reuse. num_query_words: {num_query_words}, matched_keywords: {len(matched_keywords)}, ratio: {len(matched_keywords) / num_query_words if num_query_words > 0 else 0}.")

        if should_call_curriculum_alignment:
            # This block will be entered if:
            # 1. No previous_alignment_details exist.
            # OR
            # 2. previous_alignment_details exist, but the query did not meet the heuristic conditions for reuse.
            if effective_previous_alignment_details: # Changed from previous_alignment_details
                 logger.info(f"Previous alignment details exist, but query ('{normalized_current_query}') did not meet heuristic conditions for reuse. Proceeding with new curriculum alignment call.")
            else: # No previous_alignment_details
                 logger.info(f"No previous alignment details. Proceeding with new curriculum alignment for: '{normalized_current_query}'")

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
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error decoding JSON from CurriculumAlignmentAgent: {e}. Received: {alignment_json_str}")
                # Populate with a default error structure for alignment_data to prevent downstream NoneErrors
                alignment_data = {"error": "alignment_failed", "is_in_syllabus": False, "notes_for_orchestrator": "Syllabus check failed."}
                # return True, "I had a little trouble checking the syllabus right now. Could you try asking again? exit"
        
        # Ensure alignment_data is not None before proceeding AND before storing it in agent state
        if alignment_data: # Check if alignment_data is not None (could be from reuse or new call)
            if not alignment_data.get("error"): # Only store if it's not an error alignment
                logger.debug(f"Updating Orchestrator's current_session_alignment with current turn's alignment_data: {str(alignment_data)[:200]}...")
                self.current_session_alignment = alignment_data
            else:
                logger.warning(f"Current turn's alignment_data indicates an error: {alignment_data.get('error')}. Not updating current_session_alignment with this error state.")
        else: # alignment_data is None at this point
             logger.warning("alignment_data is None for this turn (either new call failed/skipped or reuse was not applicable). Not updating current_session_alignment.")
             # Fallback if alignment_data is still None after all attempts
             if should_call_curriculum_alignment and called_curriculum_alignment_this_turn: # A call was made but resulted in None
                logger.error("Critical: alignment_data is None after a new alignment call. This should not happen.")
                alignment_data = {"error": "internal_orchestrator_error_post_call", "is_in_syllabus": False, "notes_for_orchestrator": "Internal error processing alignment after new call."}
             elif not should_call_curriculum_alignment and not alignment_data : # Reuse was intended but effective_previous_alignment_details was itself problematic or None
                logger.error("Critical: alignment_data is None after intending to reuse. This implies effective_previous_alignment_details was problematic.")
                alignment_data = {"error": "internal_orchestrator_error_post_reuse_attempt", "is_in_syllabus": False, "notes_for_orchestrator": "Internal error: failed to reuse alignment."}
             # If no call was made and no reuse was intended (e.g. empty query handled earlier), alignment_data might correctly be None if that path led here.
             # However, the logic above should have set it or it should be handled by empty query checks.
             # Adding a final safety net if it's still None for unexpected reasons:
             if alignment_data is None:
                  alignment_data = {"error": "internal_orchestrator_error_final_fallback", "is_in_syllabus": False, "notes_for_orchestrator": "Internal error: alignment data unexpectedly None."}


        reply_to_user = ""
        terminate_signal = " exit" 
        next_agent = None 
        message_for_next_agent = "" 

        if alignment_data is None:
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
                    "alignment_data": alignment_data # Alignment for the NEW subject/topic
                })
                logger.debug(f"Context switched. Routing to ConceptTutorAgent with new alignment for: '{user_query_for_specialist[:100]}...'")

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
                    "alignment_data": alignment_data # Based on the current question's alignment
                })
                logger.debug(f"Defaulting to ConceptTutorAgent. Routing to {next_agent.name}.")
            
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
                            logger.debug("Successfully parsed structured response from specialist.")
                        else:
                            # Specialist returned JSON, but not the expected dict structure
                            reply_to_user = specialist_response_summary
                            # RAG context from specialist is not in the expected format
                            logger.warning("Specialist returned JSON but not in expected answer/context format.")
                    except json.JSONDecodeError:
                        # Specialist returned a plain string, not JSON
                        reply_to_user = specialist_response_summary
                        # RAG context from specialist is not available if it wasn't JSON
                        logger.debug("Specialist returned a plain string response.")
                else:
                    reply_to_user = f"I consulted with the {next_agent.name}, but didn't get a conclusive answer. Could you rephrase?"
                    # No RAG context from specialist if no summary
                    logger.warning(f"No summary received from {next_agent.name}.")
            else:
                # This case should ideally not be reached if ConceptTutor is the default
                reply_to_user = "I've analyzed your query with the syllabus, but I'm not sure how to proceed. Can you clarify?"
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
        
        # Package the final response for main.py
        # Ensure alignment_data_to_use is defined. It would be the result from CurriculumAlignmentAgent or reused previous alignment.
        # For this change, we are ensuring that 'alignment_data' (which holds the active alignment info) is passed.
        final_orchestrator_output_payload = {
            "answer": reply_to_user, # This is the specialist's answer, or Orchestrator's direct reply
            "retrieved_rag_context_for_answer": retrieved_rag_context_for_final_reply, # RAG context associated with 'answer'
            "alignment_data_used_for_routing": alignment_data # The alignment_data used for the main decision/routing
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