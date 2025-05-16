"""
ZIMSEC Tutoring System - Student Interface Agent
------------------------------------------------

This module defines the `StudentInterfaceAgent`, responsible for managing
all direct interactions with the student (represented by `UserProxy` in Autogen).
Its main goal is to provide a friendly, supportive, and natural conversational
experience.

Key Responsibilities:
- Receives raw user input (query and form level) from the `UserProxy`.
- Packages this input into a JSON format and forwards it to the `OrchestratorAgent`.
- Receives a structured JSON response from the `OrchestratorAgent`.
  This response contains the final answer, any retrieved RAG context, and an optional
  `suggested_image_path`.
- The `StudentInterfaceAgent` itself does minimal processing of the Orchestrator's response.
  Its primary role on the return path is to ensure the response is in the correct format
  (a dictionary with "answer" and "suggested_image_path" keys) to be passed back
  to `main.py` for final presentation to the user.
- (Previously, it had more complex conversationalizing logic, but this has been simplified
  as the Orchestrator now provides a more direct payload).

Technical Details:
- Inherits from `autogen.AssistantAgent`.
- Defines a system message that emphasizes its role in friendly student interaction.
- Registers a reply function (`_generate_student_reply`) which is the core of its
  interaction logic.
- When a message is from `UserProxy` (student), it initiates a chat with the `OrchestratorAgent`.
- When it receives the reply from the `OrchestratorAgent`, it ensures it's a dictionary
  (parsing from JSON if necessary) and returns it.

Dependencies:
- autogen
- json
- typing
- logging
- OrchestratorAgent (for forwarding requests)

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""
import autogen
import json
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

# Setup logger for this module
logger = logging.getLogger(__name__)

class StudentInterfaceAgent(autogen.AssistantAgent):
    """
    The `StudentInterfaceAgent` acts as the primary point of contact with the student user.

    It is responsible for:
    1.  Receiving the student's raw input (query and selected form level) from the
        `UserProxy` agent.
    2.  Packaging this input into a structured JSON message.
    3.  Forwarding this JSON message to the `OrchestratorAgent` to initiate the main
        tutoring logic (curriculum alignment, specialist agent routing, etc.).
    4.  Receiving the final structured response from the `OrchestratorAgent`.
        This response is expected to be a JSON string which, when parsed, will contain
        at least an "answer" field and potentially a "suggested_image_path" and
        other contextual information.
    5.  Ensuring this received response is converted into a Python dictionary and
        then returning this dictionary. This dictionary is then picked up by the
        `main.py` script to be displayed to the actual user.

    The agent aims to maintain a friendly and supportive tone, as guided by its
    system message, although most of the direct conversational formatting is now
    expected to be handled by the specialist agents and refined by the Orchestrator
    before reaching this agent.
    """
    def __init__(self, name: str, orchestrator_agent, llm_config: Dict, **kwargs):
        system_message = (
            "You are the Student Interface Agent.\n"
            "Your job is to interact with the student in a friendly, supportive, and natural way.\n"
            "You receive structured responses from the Orchestrator (which may come from any subject or specialist agent).\n"
            "You must turn these into conversational, interactive messages, offering options and encouragement, and waiting for the student's reply.\n"
            "Never dump raw JSON or syllabus text. Always summarize, explain, and guide the student step by step.\n"
            "If you receive a remediation_syllabus_content response, summarize the key areas and ask the student which one they'd like to start with.\n"
            "If you receive other types of responses, do your best to explain them in a human-like way.\n"
        )
        super().__init__(name, system_message=system_message, llm_config=llm_config, **kwargs)
        self.orchestrator_agent = orchestrator_agent
        self.register_reply(
            autogen.Agent,
            StudentInterfaceAgent._generate_student_reply
        )

    @staticmethod
    def conversationalize_remediation(remediation_json: Dict) -> str:
        """
        DEPRECATED/UNUSED: Converts a JSON object containing remediation syllabus content
        into a user-friendly, conversational string with options for the student.

        This function was previously used to make raw syllabus remediation data more
        palatable for the student. The current architecture expects the Orchestrator or
        specialist agents to handle more of the conversational formatting.

        Args:
            remediation_json (Dict): A dictionary, typically from a diagnostic agent,
                                     containing 'remediation' (a list of syllabus entries)
                                     and a 'message'.

        Returns:
            str: A conversational message presenting remediation options.
        """
        remediation = remediation_json.get('remediation', [])
        message = remediation_json.get('message', '')
        if not remediation:
            return "I'm here to help! Could you tell me more about what you're struggling with?"
        intro = (
            "Let's work together to strengthen your understanding! "
            "Here are some foundational topics from earlier years that might help you:"
        )
        options = []
        for entry in remediation:
            form = entry.get('form')
            objectives = entry.get('objectives', '').split('\n')
            summary = f"**{form}**: " + ', '.join(obj.strip('- ') for obj in objectives[:2]) + ("..." if len(objectives) > 2 else "")
            options.append(summary)
        options_text = '\n'.join(f"- {opt}" for opt in options)
        followup = (
            "\nWhich of these would you like to start with? "
            "Or would you like a quick overview of all? "
            "Just let me know!"
        )
        return f"{intro}\n{options_text}\n{followup}"

    def _conversationalize_diagnostic_remediation(self, data: Dict) -> str:
        """DEPRECATED/UNUSED: Transforms diagnostic/remediation JSON into a conversational message."""
        logger.debug(f"Conversationalizing diagnostic/remediation data: {data}")
        
        if "prerequisite_check" in data and data["prerequisite_check"].get("score", 1.0) < 0.7:
            prereq = data["prerequisite_check"]
            return message
        elif "remediation_path" in data:
            path_data = data["remediation_path"]
            logger.info(f"Formatting remediation path from syllabus for topic '{path_data.get('current_topic')}'")
            
            message_parts = [f"Okay, let's look at what you need to cover for '{path_data.get('current_topic')} / {path_data.get('current_subtopic')}' from previous forms."]
            
            return "\n".join(message_parts)
        elif "score" in data and "status" in data:
            logger.info(f"Formatting standard diagnostic score. Score: {data['score']}, Status: {data['status']}")
            if data["status"] == "mastered":
                return f"Great job! It looks like you've mastered this. Your score is {data['score']:.2f}."
            return message
        elif "error" in data:
            logger.warning(f"Formatting error message: {data['error']}")
            return f"I encountered an issue: {data['error']}. Could you try rephrasing or asking something else?"
        else:
            logger.warning(f"Unhandled diagnostic/remediation data structure: {data}")
            return "I received a response, but I'm not sure how to present it. It might be a bit technical."

    def _conversationalize_concept_explanation(self, explanation_text: str) -> str:
        """DEPRECATED/UNUSED: Wraps a concept explanation in a conversational tone."""
        logger.debug(f"Conversationalizing concept explanation snippet: {explanation_text[:100]}...")
        # Basic conversational wrapper. Could be more dynamic.
        return f"Okay, let me explain that for you:\n\n{explanation_text}\n\nDoes that make sense, or would you like me to explain a part of it differently?"

    async def _generate_student_reply(self, messages: Optional[List[Dict]] = None, sender: Optional[autogen.Agent] = None, config: Optional[Any] = None) -> Tuple[bool, Union[str, Dict, None]]:
        """
        Handles messages for the StudentInterfaceAgent.

        This asynchronous method is the main interaction point for this agent.
        Its behavior depends on the sender of the message:

        1.  If the message is from `UserProxy` (i.e., the student via the main CLI):
            a.  Parses the incoming content (expected to be a JSON string with
                `user_query` and `form_level`).
            b.  Constructs an input JSON string for the `OrchestratorAgent`.
            c.  Initiates a chat with the `OrchestratorAgent`, sending this input.
            d.  Receives the reply from the `OrchestratorAgent` (which should be a
                JSON string representing a dictionary with "answer", "suggested_image_path", etc.).
            e.  Parses this orchestrator reply into a Python dictionary.
            f.  Returns this dictionary. This dictionary is then used by `main.py`
                to display the answer and image path to the student.

        2.  If the message is from another agent (e.g., directly from Orchestrator or a
            specialist, though this path is less common in the primary student interaction loop):
            a.  It attempts to parse the content as JSON if it's a string.
            b.  Returns the parsed content (if successful) or a wrapped version if parsing fails,
                maintaining a consistent dictionary structure with "answer" and
                "suggested_image_path".

        Args:
            messages (Optional[List[Dict]]): A list of messages. The last message is the one to process.
            sender (Optional[autogen.Agent]): The agent that sent the message.
            config (Optional[Any]): Optional configuration data (not actively used).

        Returns:
            Tuple[bool, Union[str, Dict, None]]: A tuple where the first element is True
                                                 (indicating the agent can reply), and the
                                                 second is the processed reply (typically a
                                                 dictionary for `main.py` or None).
        """
        last_message = messages[-1]
        content = last_message.get("content", "")
        logger.debug(f"Received message from {sender.name if sender else 'Unknown Sender'}")
        
        # If this is a student message, forward to orchestrator
        if sender.name == "UserProxy":
            # Parse the JSON and extract the user_query and form_level
            try:
                data = json.loads(content)
                user_query = data.get("user_query", content)
                form_level = data.get("form_level", "Form 4")
                orchestrator_input = json.dumps({"user_query": user_query, "form_level": form_level})
            except Exception:
                orchestrator_input = content
            logger.debug(f"Forwarding to Orchestrator: {orchestrator_input}")
            orchestrator_results = await self.a_initiate_chat(
                recipient=self.orchestrator_agent,
                message=orchestrator_input,
                max_turns=1,
                summary_method="last_msg",
                silent=False
            )
            orchestrator_reply = orchestrator_results.summary if orchestrator_results else "Sorry, I couldn't get a response."
            # Try to parse as JSON
            try:
                parsed_orchestrator_reply = json.loads(orchestrator_reply)
                # If orchestrator_reply was valid JSON, pass the parsed dict as the reply.
                # main.py will then extract 'answer', 'suggested_image_path', etc.
                reply = parsed_orchestrator_reply
            except json.JSONDecodeError:
                # If orchestrator_reply was not JSON, pass the raw string.
                # Create a simple dict structure for consistency in main.py if needed.
                reply = {"answer": orchestrator_reply, "suggested_image_path": None}
            except Exception as e: # Catch other potential errors during parsing or access
                logger.error(f"Error processing orchestrator reply: {e}. Raw reply: {orchestrator_reply}")
                reply = {"answer": "Sorry, I had trouble processing the response.", "suggested_image_path": None}
            return True, reply
        else:
            # If this is a specialist/orchestrator reply directly to StudentInterfaceAgent (not via UserProxy flow),
            # this case should ideally not be hit in the main loop for student interaction.
            # For safety, pass content through, perhaps wrapped for consistency.
            if isinstance(content, str):
                try:
                    parsed_content = json.loads(content)
                    return True, parsed_content # Assuming it's already the final format
                except json.JSONDecodeError:
                    return True, {"answer": content, "suggested_image_path": None}
            return True, content # Pass as is if already a dict or other type 