import autogen
import json
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

# Setup logger for this module
logger = logging.getLogger(__name__)

class StudentInterfaceAgent(autogen.AssistantAgent):
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
        """Transforms diagnostic/remediation JSON into a conversational message."""
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
        """Wraps a concept explanation in a conversational tone."""
        logger.debug(f"Conversationalizing concept explanation snippet: {explanation_text[:100]}...")
        # Basic conversational wrapper. Could be more dynamic.
        return f"Okay, let me explain that for you:\n\n{explanation_text}\n\nDoes that make sense, or would you like me to explain a part of it differently?"

    async def _generate_student_reply(self, messages: Optional[List[Dict]] = None, sender: Optional[autogen.Agent] = None, config: Optional[Any] = None) -> Tuple[bool, Union[str, Dict, None]]:
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
                parsed = json.loads(orchestrator_reply)
                if parsed.get("status") == "remediation_syllabus_content":
                    reply = self.conversationalize_remediation(parsed)
                else:
                    # Fallback: just show the message or a generic explanation
                    reply = parsed.get("message") or str(parsed)
            except Exception:
                reply = orchestrator_reply
            return True, reply
        else:
            # If this is a specialist/orchestrator reply, just pass it through (shouldn't happen in normal flow)
            return True, content 