# concept_tutor_agent.py

import autogen
import json
import asyncio # Added for async function
import os
import logging # <--- Add this import

# Setup logger for this module
logger = logging.getLogger(__name__) # <--- Add this line

# Import the RAG integration function using absolute path
# from ..rag_integration import get_knowledge_content_from_rag # OLD
from zimsec_tutor_agent_system.rag_integration import get_knowledge_content_from_rag # NEW

class ConceptTutorAgent(autogen.AssistantAgent):
    def __init__(self, name, llm_config, **kwargs):
        system_message = (
            "You are the Concept Tutor Agent in a multi-agent AI tutoring system for ZIMSEC O-Level students.\n"
            "You will receive a JSON message containing the user's original query and detailed syllabus alignment context from the Orchestrator Agent.\n"
            "Your primary goal is to provide a clear, step-by-step explanation for the user's query, grounded in the provided context.\n"
            "Follow these steps:\n"
            "1. Acknowledge the specific topic based on the context (e.g., 'Okay, let's look at finding the median for grouped data.').\n"
            "2. Use the provided syllabus outcomes and mandatory terms to guide your explanation.\n"
            "3. Incorporate the retrieved knowledge content (which will be provided in the prompt context) to explain the concept or procedure.\n"
            "4. Start with a simple definition or the first key step.\n"
            "5. Consider asking a brief Socratic question to engage the learner if appropriate (e.g., 'Before we use the formula, what does cumulative frequency tell us?').\n"
            "6. Present information clearly using Markdown formatting (bold terms, lists, code blocks for math if needed).\n"
            "7. Keep the explanation focused on the specific query and syllabus outcomes.\n"
            "8. Ensure you use the key `mandatory_terms` from the alignment data in your explanation.\n"
            "9. If the user's query is a direct request to 'calculate', 'find the value of', 'compute', or similar, and all necessary numerical inputs are provided or are simple to deduce from the context and retrieved knowledge, perform the calculation. Clearly show the method, the steps involved, and provide the final numerical answer as part of your response."
        )
        super().__init__(name, system_message=system_message, llm_config=llm_config, **kwargs)
        
        self.register_reply(
            autogen.Agent, # Triggered by messages from other agents (Orchestrator)
            ConceptTutorAgent._generate_tutor_reply
        )

    # Remove or comment out the mock RAG function
    # def _mock_retrieve_knowledge_content(self, topic: str, subtopic: str) -> str:
    #     # ... mock implementation ...

    async def _generate_tutor_reply(self, messages, sender, config):
        """Generates the explanation based on context from Orchestrator."""
        last_message = messages[-1]
        message_content = last_message.get("content", "")

        try:
            orchestrator_data = json.loads(message_content)
            user_query = orchestrator_data.get("original_user_query", "")
            alignment_data = orchestrator_data.get("alignment_data", {})
            logger.debug(f"\n ConceptTutor: Received data from Orchestrator for query '{user_query}'")
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"\n ConceptTutor: Error decoding JSON from Orchestrator: {e}")
            return True, "I received the request, but had trouble understanding the details... TERMINATE"

        topic = alignment_data.get("identified_topic", "Unknown Topic")
        subtopic = alignment_data.get("identified_subtopic", "Unknown Subtopic")
        # Get the form from alignment data, crucial for knowledge retrieval
        form = alignment_data.get("identified_form", "Form 4") # Default to Form 4 if not present
        outcomes = alignment_data.get("matched_outcomes", [])
        mandatory_terms = alignment_data.get("mandatory_terms", [])

        # Retrieve knowledge content using actual RAG function
        logger.debug(f"\n ConceptTutor: Retrieving knowledge using RAG for Topic='{topic}', Subtopic='{subtopic}', Form='{form}'")
        retrieved_content_str = get_knowledge_content_from_rag(topic, subtopic, form)

        # This is crucial: if RAG fails or returns no content, we should still inform the LLM.
        if not retrieved_content_str: # Check if the string is empty or None
            retrieved_content_str = "No specific knowledge content was retrieved for this topic. Please answer based on general knowledge if possible, or indicate that you cannot provide a specific explanation without more information."
            logger.warning(f"No RAG content retrieved for Topic='{topic}', Subtopic='{subtopic}', Form='{form}'")
        else:
            logger.debug(f"RAG content prepared for LLM. Snippet: {retrieved_content_str[:200]}...")

        # Construct a new prompt for the LLM containing all context
        # Note: This message history is internal to this agent's LLM call
        internal_messages = [
            {"role": "system", "content": self.system_message}, # Remind LLM of its persona/goal
            {
                "role": "user", 
                "content": f"Okay, I need to explain the following user query: '{user_query}'.\\n\\n" \
                           f"The syllabus context is:\\n" \
                           f"- Topic: {topic}\\n" \
                           f"- Subtopic: {subtopic}\\n" \
                           f"- Relevant Outcomes: {', '.join(outcomes) if outcomes else 'N/A'}\\n" \
                           f"- Key Mandatory Terms to use: {', '.join(mandatory_terms) if mandatory_terms else 'N/A'}\\n\\n" \
                           f"Here is relevant knowledge content I retrieved:\\n---\\n{retrieved_content_str}\\n---\\n\\n" \
                           f"Considering the user's query: '{user_query}' and your system prompt instructions (especially step 9 if it's a calculation request with given values), " \
                           f"please generate a comprehensive response. If it's a calculation, explain the method, show the steps, and provide the final numerical answer. " \
                           f"Otherwise, provide a clear explanation of the concept. Use Markdown for formatting."
            }
        ]
        
        # Generate the reply using the AssistantAgent's standard method
        # This uses the llm_config passed during initialization
        logger.debug(f"\n ConceptTutor: Generating LLM reply based on context...")
        # Use the agent's inherited method to generate reply based on internal messages
        # This requires access to the client, often done via self.client
        success, reply_obj = await self.a_generate_oai_reply(internal_messages)

        reply_content_str = None
        if success:
            if isinstance(reply_obj, str):
                reply_content_str = reply_obj
            elif isinstance(reply_obj, dict):
                # Extract content if it's a dict (common structure)
                reply_content_str = reply_obj.get("content")
                logger.debug(f"Extracted content from structured reply.")
            else:
                 logger.warning(f"Received unexpected reply format: {type(reply_obj)}. Reply: {reply_obj}")

        if reply_content_str:
            logger.debug(f"Generated Explanation Snippet: {reply_content_str[:200]}...")
            # Package the answer and the RAG context used
            final_payload = {
                "answer": reply_content_str,
                "retrieved_rag_context": retrieved_content_str # This is the context from get_knowledge_content_from_rag
            }
            return True, json.dumps(final_payload)
        else:
            logger.error(f"Failed to generate LLM reply or extract content. Success={success}, Reply Obj={reply_obj}")
            # Even on failure, attempt to return the structure if some RAG context was retrieved
            error_payload = {
                "answer": "I understand the request, but I'm having trouble formulating an explanation right now. TERMINATE",
                "retrieved_rag_context": retrieved_content_str if retrieved_content_str else "No RAG context available due to error."
            }
            return True, json.dumps(error_payload)

if __name__ == '__main__':
    # Example for testing the agent in isolation
    config_list_test = [
        {
            "model": "gpt-4.1-nano",
            "api_key": os.environ.get("OPENAI_API_KEY")
        }
    ]

    concept_tutor = ConceptTutorAgent(
        name="ConceptTutorTest",
        llm_config={"config_list": config_list_test}
    )

    # Mock message from Orchestrator
    mock_orchestrator_message = {
        "original_user_query": "Tell me about the median from a grouped frequency table.",
        "alignment_data": {
            "is_in_syllabus": True,
            "alignment_score": 0.95,
            "matched_outcomes": ["F4-Stats-MedianGrouped-Obj1", "F4-Stats-CFcurve-Obj3"],
            "mandatory_terms": ["median class", "cumulative frequency", "interpolation", "lower class boundary"],
            "identified_subject": "Mathematics",
            "identified_topic": "Statistics",
            "identified_subtopic": "Measures of Central Tendency (Grouped Data)",
            "identified_form": "Form 4" # Ensure form is part of alignment_data for RAG
        }
    }

    # Simulate receiving message (content should be structured)
    # We might need the LLM to process this structured input based on the system prompt
    logger.debug("--- Testing Concept Tutor Agent (LLM Reply) ---")
    # Use a dummy sender for the test
    dummy_sender = autogen.Agent(name="DummyOrchestrator")
    
    # Autogen works best with string messages, so pass JSON as string
    # This is an async function, so we need to run it in an event loop for testing
    async def run_concept_tutor_test():
        success_test, reply_test = await concept_tutor._generate_tutor_reply(
            messages=[{"role": "user", "content": json.dumps(mock_orchestrator_message)}],
            sender=dummy_sender,
            config=None
        )
        if success_test:
            logger.info(f"Concept Tutor Test Reply:\n{reply_test}")
        else:
            logger.error("Concept Tutor Test Failed.")

    if os.environ.get("OPENAI_API_KEY"):
         asyncio.run(run_concept_tutor_test())
    else:
        logger.warning("OPENAI_API_KEY not set. Skipping ConceptTutorAgent __main__ test.")
    
    logger.debug("Concept Tutor agent created. Test execution in isolation needs refinement or run via main.py.") 