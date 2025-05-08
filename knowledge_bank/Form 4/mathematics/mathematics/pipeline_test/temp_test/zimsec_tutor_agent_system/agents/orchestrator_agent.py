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

class OrchestratorAgent(autogen.AssistantAgent):
    def __init__(self, name, llm_config, curriculum_alignment_agent, concept_tutor_agent, **kwargs):
        system_message = (
            "You are the Orchestrator Agent in a multi-agent AI tutoring system for ZIMSEC O-Level students.\n"
            "Your primary roles are:\n"
            "1. Receive the initial user query from the UserProxy.\n"
            "2. ALWAYS first consult the CurriculumAlignmentAgent. You will send it a JSON message like: {\"user_query\": \"<actual_user_query>\", \"learner_context\": {\"current_subject_hint\": \"<current_subject_or_default>\"}}.\n"
            "3. Receive and parse the JSON response from the CurriculumAlignmentAgent.\n"
            "4. Based on the alignment details, decide the next step and formulate a response or action.\n"
            "   - If out_of_scope, inform the user politely.\n"
            "   - If relevant to a different subject, clarify with the user.\n"
            "   - If aligned, (for now) state this and that you would next determine intent and route to a specialist.\n"
            "Communication Guidelines with CurriculumAlignmentAgent:\n"
            "- Your message TO the CurriculumAlignmentAgent MUST be a JSON string.\n"
            "- You will receive a JSON string as a response FROM the CurriculumAlignmentAgent.\n"
            "Your reply in the main chat will be your analysis and intended next steps, directly addressing the UserProxy."
            "Key new responsibilities:\n"
            "- After asking the user to clarify subject context, await their response (SIMULATED FOR NOW).\n"
            "- If they confirm a switch, acknowledge and (conceptually) prepare to route to a specialist (e.g., ConceptTutor) with the new context.\n"
            "- If they decline or ask a new question, re-initiate curriculum alignment for the new query.\n"
            "- When concluding your turn based on current logic (e.g., after alignment check), end your message with \"exit\" to signal termination to the UserProxy in test mode."
            "Key responsibilities update:\n"
            "- If query is aligned with current context: Route the query and alignment details to the ConceptTutorAgent.\n"
            "- Your final reply to the UserProxy should be the response received from the ConceptTutorAgent."
        )
        super().__init__(name, system_message=system_message, llm_config=llm_config, **kwargs)
        self.curriculum_alignment_agent = curriculum_alignment_agent
        self.concept_tutor_agent = concept_tutor_agent # Store the concept tutor agent
        # self._conversation_state = {} # State management removed for simpler simulation
        
        # Register the main reply generation function
        self.register_reply(
            autogen.Agent, # Trigger for messages from any other agent (typically UserProxy)
            OrchestratorAgent._generate_orchestrator_reply
        )

    async def _generate_orchestrator_reply(self, messages, sender, config):
        """
        This function is called when the OrchestratorAgent receives a message.
        It processes the user query, consults curriculum alignment, and decides next steps.
        """
        last_message = messages[-1]
        user_input = last_message.get("content", "")
        # conversation_id = messages[0].get("id") # State management removed for simulation

        print(f"\nOrchestrator: Received input: '{user_input}' from {sender.name}")

        # Handle empty input gracefully if it somehow gets through
        if not user_input.strip():
             print("\nOrchestrator: Received empty input. Ending turn.")
             return True, "Looks like there was no question. Ask me something else! exit"

        print(f"\nOrchestrator: Handling as initial query: '{user_input}'")
        learner_context = {"current_subject_hint": "Mathematics"} # Default context
        message_for_alignment_agent = json.dumps({
            "user_query": user_input,
            "learner_context": learner_context
        })
        
        print(f"\nOrchestrator: Sending to CurriculumAlignmentAgent: {message_for_alignment_agent}")

        chat_results = await self.a_initiate_chat(
            recipient=self.curriculum_alignment_agent,
            message=message_for_alignment_agent,
            max_turns=1, 
            summary_method="last_msg",
            silent=False # Let's see the sub-chat for debugging
        )
        alignment_json_str = chat_results.summary

        print(f"\nOrchestrator: Received from CurriculumAlignmentAgent (raw): {alignment_json_str}")

        try:
            alignment_data = json.loads(alignment_json_str)
            print(f"\nOrchestrator: Parsed alignment data: {json.dumps(alignment_data, indent=2)}")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"\nOrchestrator: Error decoding JSON: {e}. Received: {alignment_json_str}")
            # Add termination signal even to error messages
            return True, "I had a little trouble checking the syllabus right now. Could you try asking again? exit"

        reply_to_user = ""
        terminate_signal = " exit" # Add space for clarity before keyword

        if alignment_data is None:
             reply_to_user = "I couldn't get syllabus alignment information. Please try again."
        elif alignment_data.get("error") == "out_of_scope" or not alignment_data.get("is_in_syllabus"):
            reply_to_user = "It seems your question might be outside the scope of the ZIMSEC O-Level syllabus I cover."
            if alignment_data.get("notes_for_orchestrator"):
                 reply_to_user += f" ({alignment_data.get('notes_for_orchestrator')})"
            reply_to_user += " Do you have a question related to ZIMSEC Mathematics or Combined Science?"
        elif alignment_data.get("identified_subject") and alignment_data.get("identified_subject") != learner_context.get("current_subject_hint"):
            identified_subject = alignment_data.get('identified_subject')
            topic_info = f"{alignment_data.get('identified_topic', 'N/A')} / {alignment_data.get('identified_subtopic', 'N/A')}"
            question_to_user = (
                f"That's an interesting question! It looks like it aligns best with **{identified_subject}** under the topic: *{topic_info}*. "
                f"Our current context is {learner_context.get('current_subject_hint')}. "
                f"Would you like to switch our focus to {identified_subject} to discuss this? (yes/no)"
            )
            print(f"\nOrchestrator would ask user: {question_to_user}")
            
            # ------ SIMULATION START: Simulate user replying 'yes' ------
            print("\nOrchestrator: SIMULATING user replied 'yes' to context switch.")
            simulated_user_reply = "yes"
            # This is where the logic from the other branch of the original stateful reply handler goes:
            if simulated_user_reply.lower() in ["yes", "ok", "sure", "yep", "please do", "switch"]:
                # CONCEPTUAL NEXT STEP: Update context, re-align, then route to Concept Tutor for the *original* query
                reply_to_user = f"Great! Switching context to **{identified_subject}**...\n(Next: Routing to ConceptTutorAgent for '{user_input[:30]}...')"
                # In a real implementation: update context, re-align, route.
            else:
                 # This part of simulation won't be reached if we simulate "yes"
                 reply_to_user = "Okay, we'll stick to the current context. Do you have another question?"
            # ------ SIMULATION END ------
            
        else:
            # --- Query IS Aligned with Current Context --- 
            subject = alignment_data.get('identified_subject', 'the syllabus')
            topic_info = f"{alignment_data.get('identified_topic', 'N/A')} / {alignment_data.get('identified_subtopic', 'N/A')}"
            print(f"\nOrchestrator: Query aligned with {subject} - Topic: {topic_info}. Routing to ConceptTutorAgent.")

            # Prepare message for ConceptTutorAgent
            message_for_tutor = json.dumps({
                "original_user_query": user_input,
                "alignment_data": alignment_data
            })

            # Initiate chat with ConceptTutorAgent
            # The Orchestrator initiates this chat and waits for the reply.
            tutor_chat_results = await self.a_initiate_chat(
                recipient=self.concept_tutor_agent,
                message=message_for_tutor,
                max_turns=1, # Expect one reply from the tutor for now
                summary_method="last_msg",
                silent=False # Let's see the tutor's reply
            )
            reply_from_tutor = tutor_chat_results.summary
            
            # The tutor's reply becomes the reply to the user
            if reply_from_tutor:
                reply_to_user = reply_from_tutor
                terminate_signal = "" # Let the tutor control termination if needed later, remove orchestrator's exit
            else:
                # Handle cases where the tutor fails to reply
                print("\nOrchestrator: ConceptTutorAgent did not provide a reply.")
                reply_to_user = "I understand the topic is aligned, but I had trouble getting an explanation ready. Please try asking again."
                terminate_signal = " exit" # Terminate if tutor failed

        # Append termination signal ONLY if we didn't get a reply from the tutor
        final_reply = reply_to_user + terminate_signal 
        print(f"\nOrchestrator responding to UserProxy: {final_reply}")
        return True, final_reply

if __name__ == '__main__':
    # This testing block is more complex now as Orchestrator initiates chats.
    # It's better to test this interaction within the main_script.py setup.
    # However, a simplified direct call might look like this if other agents are minimal.
    
    print("OrchestratorAgent self-test (conceptual - real test via main.py recommended)")
    
    # Minimal config for testing agent instantiation
    config_list_test = [
        {"model": "phi4:latest", "api_key": "ollama", "base_url": "http://localhost:11434/v1", "api_type": "ollama"}
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
        concept_tutor_agent=curriculum_agent_for_test
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