import asyncio
import autogen
import json
import sys
import os
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import time
import logging

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zimsec_tutor_agent_system.agents import (
    CurriculumAlignmentAgent,
    OrchestratorAgent,
    ConceptTutorAgent,
    DiagnosticRemediationAgent,
    AssessmentRevisionAgent,
    ProjectsMentorAgent,
    ContentGenerationAgent,
    AnalyticsProgressAgent,
    StudentInterfaceAgent
)
from memory_manager import ConversationMemory # Assuming memory_manager.py is in the same directory or accessible via PYTHONPATH

# Initialize Typer app and Rich console
app = typer.Typer()
console = Console()

# Available form levels
FORM_LEVELS = ["Form 1", "Form 2", "Form 3", "Form 4"]

# Helper function to log to file and print to Rich console
def log_and_print(content, level=logging.DEBUG, log_prefix=""):
    """Logs the content to the configured logger and prints to Rich console."""
    global console
    
    log_message = ""
    if isinstance(content, str):
        log_message = content
    elif hasattr(content, 'renderable') and hasattr(content.renderable, 'markup'): # e.g. Panel(Markdown(...))
        log_message = content.renderable.markup
    elif hasattr(content, 'renderable') and isinstance(content.renderable, str): # e.g. Panel("string")
        log_message = content.renderable
    elif hasattr(content, 'markup'): # For Markdown
        log_message = content.markup
    elif hasattr(content, 'title') and isinstance(content.title, str): # For Panel, use title if renderable is complex
         log_message = f"Panel: {content.title}"
    else:
        try:
            # Attempt to convert common Rich objects to a string representation
            if hasattr(content, 'plain'):
                log_message = content.plain
            elif hasattr(content, 'text') and isinstance(content.text, list) and len(content.text) > 0:
                 log_message = str(content.text[0]) if isinstance(content.text[0], str) else str(content.text)
            else:
                log_message = str(content)
        except Exception:
            log_message = "Rich object (unable to stringify for plain log)"

    # Remove ANSI escape codes for cleaner logs, if any made it through string conversion
    import re
    ansi_escape = re.compile(r'\\x1B(?:[@-Z\\\\-_]|[[0-?]*[ -/]*[@-~])')
    log_message = ansi_escape.sub('', log_message)

    logging.log(level, f"{log_prefix}{log_message.strip()}")
    console.print(content)

# Shared config_list for Ollama models
config_list_openai_ollama = [
    {
        "model": "gpt-4.1-nano",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }
]

# Config list for ProjectsMentorAgent
config_list_openai_gpt_for_mentor = [
    {
        "model": "gpt-4.1-nano",
        "api_key": os.environ.get("OPENAI_API_KEY")
    }
]

# Define the schema for the project_checklist tool
PROJECT_CHECKLIST_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "project_checklist",
        "description": "Provides a checklist or guidance for a given CALA project milestone.",
        "parameters": {
            "type": "object",
            "properties": {
                "milestone": {
                    "type": "string",
                    "description": "The project milestone (e.g., 'plan', 'investigation', 'solution_dev', 'presentation', 'evaluation').",
                },
                "draft_snippet": {
                    "type": "string",
                    "description": "A snippet of the student's current draft for context.",
                },
            },
            "required": ["milestone"], 
        },
    },
}

def initialize_agents():
    """Initialize all agents with their configurations"""
    # 1. Initialize CurriculumAlignmentAgent
    curriculum_agent = CurriculumAlignmentAgent(
        name="CurriculumAlignmentAgent",
        llm_config={"config_list": config_list_openai_ollama}
    )

    # 2. Initialize all specialist agents
    concept_tutor = ConceptTutorAgent(
        name="ConceptTutorAgent",
        llm_config={"config_list": config_list_openai_ollama}
    )

    diagnostic_agent = DiagnosticRemediationAgent(
        name="DiagnosticRemediationAgent",
        llm_config={"config_list": config_list_openai_ollama}
    )

    assessment_agent = AssessmentRevisionAgent(
        name="AssessmentRevisionAgent",
        llm_config={"config_list": config_list_openai_ollama}
    )

    projects_mentor_agent = ProjectsMentorAgent(
        name="ProjectsMentorAgent",
        llm_config={
                    "config_list": config_list_openai_gpt_for_mentor,
            "tools": [PROJECT_CHECKLIST_TOOL_SCHEMA]
                },
                code_execution_config={"last_n_messages": 2, "use_docker": False}
    )

    content_generation_agent = ContentGenerationAgent(
        name="ContentGenerationAgent",
        llm_config={"config_list": config_list_openai_ollama}
    )

    analytics_progress_agent = AnalyticsProgressAgent(
        name="AnalyticsProgressAgent",
        llm_config={"config_list": config_list_openai_ollama}
    )

    # 3. Initialize OrchestratorAgent
    orchestrator_agent = OrchestratorAgent(
        name="OrchestratorAgent",
        llm_config={"config_list": config_list_openai_ollama},
                code_execution_config={"last_n_messages": 3, "use_docker": False},
        curriculum_alignment_agent=curriculum_agent,
        concept_tutor_agent=concept_tutor,
        diagnostic_agent=diagnostic_agent,
        assessment_agent=assessment_agent,
        projects_mentor_agent=projects_mentor_agent,
        content_generation_agent=content_generation_agent,
        analytics_progress_agent=analytics_progress_agent
    )

    # 4. Initialize StudentInterfaceAgent
    student_interface_agent = StudentInterfaceAgent(
        name="StudentInterfaceAgent",
        orchestrator_agent=orchestrator_agent,
        llm_config={"config_list": config_list_openai_ollama}
    )

    return student_interface_agent

async def interactive_session(debug: bool = False):
    """Run an interactive session with the tutoring system"""
    # Initialize agents
    student_interface_agent = initialize_agents()
    
    # Initialize UserProxy
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
                human_input_mode="TERMINATE",
                is_termination_msg=lambda x: x.get("content", "").strip().lower() in ["exit", "quit"],
        code_execution_config=False,
    )

    # Initialize ConversationMemory
    # openai_api_key = os.getenv("OPENAI_API_KEY") # No longer needed for local embeddings
    # conversation_memory = ConversationMemory(openai_api_key=openai_api_key) # Incorrect instantiation
    
    # Instantiate with defaults for local Ollama embedding model (nomic-embed-text)
    # This uses the defaults defined in ConversationMemory (nomic-embed-text, http://localhost:11434)
    conversation_memory = ConversationMemory()
    
    # If your Ollama is running elsewhere or you want a different model from the default, 
    # you would uncomment and modify the lines below:
    # conversation_memory = ConversationMemory(
    #    embedding_model_name="your-other-local-model-name", # e.g., "mxbai-embed-large"
    #    ollama_base_url="http://your-ollama-host:port"  # e.g., "http://192.168.1.100:11434"
    # )
    current_student_id = "student_001" # Example student ID

    # Welcome message
    log_and_print(Panel(
        "[bold green]Welcome to the ZIMSEC Tutoring System![/bold green]\n"
        "You can ask questions about mathematics, and I'll help you learn.\n"
        "Type 'exit' or 'quit' to end the session.",
        title="Interactive Session",
        border_style="blue"
    ))

    # Ask for form level
    form = ""
    while form not in FORM_LEVELS:
        form = typer.prompt("Enter your form level (Form 1-4)", default="Form 4")
        if form not in FORM_LEVELS:
            log_and_print(f"[bold red]Invalid form level: {form}[/bold red]", level=logging.WARNING)
            log_and_print(f"[bold yellow]Available form levels: {', '.join(FORM_LEVELS)}[/bold yellow]", level=logging.INFO)

    log_and_print(f"[bold green]Using form level:[/bold green] {form}")
    log_and_print("[bold]You can now ask your questions. Type 'exit' or 'quit' to end.[/bold]")

    retrieved_rag_context_for_current_turn = None
    last_orchestrator_response_payload = None # Initialize here

    while True:
        # Get user question
        question = typer.prompt("\nYour question")
        
        if question.lower() in ["exit", "quit"]:
            log_and_print("[bold green]Goodbye![/bold green]")
            break

        # Initialize augmented_question with the raw question by default
        # This ensures it's always defined before being used in payload_to_student_interface
        augmented_question = question

        # 1. Determine if context should be reset (now an async call)
        should_reset, reset_reason = await conversation_memory.should_reset_context(current_student_id, question)
        if should_reset:
            conversation_memory.clear_history(current_student_id)
            log_and_print(f"[italic yellow]System: New conversation started. Reason: {reset_reason}[/italic yellow]", level=logging.DEBUG)
            retrieved_rag_context_for_current_turn = None
            # For a new conversation, augmented_question remains the raw question (already set)
        else:
            log_and_print(f"[italic green]System: Continuing conversation for {current_student_id}.[/italic green]", level=logging.DEBUG)
            # Augment the question with history
            interaction_history = conversation_memory.history.get(current_student_id, [])
            last_interaction = interaction_history[-1] if interaction_history else None
            
            # augmented_question = question # No longer needed here, initialized outside
            previous_rag_context_for_memory = None # This is for memory prompt augmentation
            previous_response_for_memory = None

            if last_interaction:
                previous_question_for_memory = last_interaction.user_query
                previous_response_for_memory = last_interaction.agent_response
                previous_rag_context_for_memory = last_interaction.retrieved_context # Context that was part of the last answer

                augmented_question_parts = []
                if previous_question_for_memory:
                    augmented_question_parts.append(f"Previous Question: {previous_question_for_memory}")
                # We need to ensure that the agent_response (which is a JSON string from orchestrator) is handled correctly
                # For the specialist agent, the 'answer' part is the actual text it should see as previous response.
                if previous_response_for_memory: # This is the JSON string like {"answer": ..., "retrieved_rag_context_for_answer": ...}
                    try:
                        prev_resp_payload = json.loads(previous_response_for_memory)
                        actual_prev_answer = prev_resp_payload.get("answer", previous_response_for_memory)
                        # The RAG context for the specialist is implicitly part of the previous answer if structured that way.
                        # For the specialist, we give it the actual text of the previous answer.
                        augmented_question_parts.append(f"Previous Response: {actual_prev_answer}") 
                        # If the previous RAG context was used for that answer, it should be presented too.
                        if previous_rag_context_for_memory: # This is the context used for the previous answer
                             augmented_question_parts.append(f"Previous Context Used: {previous_rag_context_for_memory}")
                    except json.JSONDecodeError:
                        augmented_question_parts.append(f"Previous Response: {previous_response_for_memory}") # Fallback if not JSON
                
                augmented_question_parts.append(f"Student's Current Question: {question}")
                augmented_question = "\\n".join(augmented_question_parts)

        payload_to_student_interface = {
            "user_query": augmented_question, # This is now potentially augmented
            "original_user_query": question, # Always the raw current question
            "form_level": form,
            "context_was_reset": should_reset
        }

        # If continuing and we have previous orchestrator output, pass previous alignment
        if not should_reset and last_orchestrator_response_payload:
            prev_alignment = last_orchestrator_response_payload.get('alignment_data_used_for_routing')
            if prev_alignment:
                payload_to_student_interface['previous_alignment_details'] = prev_alignment
                log_and_print(f"[italic blue]System: Passing previous alignment details to Orchestrator.[/italic blue]", level=logging.DEBUG)

        # 3. Initiate chat with StudentInterfaceAgent, which will then talk to Orchestrator
        # StudentInterfaceAgent acts as a layer that could, in the future, do more complex student modeling or interface tasks.
        message = json.dumps(payload_to_student_interface)

        # Show debug info if enabled
        if debug:
            log_and_print("\n[bold yellow]Debug Mode:[/bold yellow]", level=logging.DEBUG)
            log_and_print(f"Question: {question}", level=logging.DEBUG)
            log_and_print(f"Form Level: {form}", level=logging.DEBUG)
            log_and_print("[bold]Processing...[/bold]", level=logging.DEBUG)

        # Get response from the system
        result = await user_proxy.a_initiate_chat(
            recipient=student_interface_agent,
            message=message,
            max_turns=1
        )

        # Debug: Show the full result object if in debug mode
        if debug:
            log_and_print("\n[bold yellow]Debug Information:[/bold yellow]", level=logging.DEBUG)
            log_and_print(Panel(str(result), title="Raw Result", border_style="yellow"), level=logging.DEBUG)
        
        # Extract and display the actual response
        final_agent_response_text = "Default response" # Initialize
        actual_rag_context_used_this_turn = None # Initialize

        try:
            last_message_content = None
            if result and result.chat_history:
                for msg in reversed(result.chat_history):
                    if msg.get('name') != user_proxy.name and msg.get('content') is not None:
                        last_message_content = msg['content']
                        break
            
            if isinstance(last_message_content, str):
                try:
                    parsed_content = json.loads(last_message_content)
                    if isinstance(parsed_content, dict):
                        final_agent_response_text = parsed_content.get("answer", str(parsed_content))
                        actual_rag_context_used_this_turn = parsed_content.get("retrieved_rag_context")
                        last_orchestrator_response_payload = parsed_content # Update for NEXT turn
                    else:
                        final_agent_response_text = last_message_content
                        last_orchestrator_response_payload = None # Reset if not a dict
                except json.JSONDecodeError:
                    final_agent_response_text = last_message_content
                    last_orchestrator_response_payload = None # Reset if not JSON
            elif isinstance(last_message_content, dict): # e.g. tool_calls, not our structured response
                final_agent_response_text = json.dumps(last_message_content) # Or handle more gracefully
                last_orchestrator_response_payload = None # Reset as it's not the expected payload
            elif result and result.summary: # Fallback to summary if no suitable chat history message
                 final_agent_response_text = result.summary
                 last_orchestrator_response_payload = None # Reset
            elif not last_message_content: # If no suitable message found in history
                final_agent_response_text = "No specific response content found in chat history."
                last_orchestrator_response_payload = None # Reset

            if not final_agent_response_text and result: # Broader fallback
                 final_agent_response_text = str(result) if result else "I didn't receive a response. Please try again."
                 last_orchestrator_response_payload = None # Reset

            # Clean up the response
            if isinstance(final_agent_response_text, str):
                for term in ["exit", "quit", "TERMINATE"]:
                    final_agent_response_text = final_agent_response_text.replace(term, "").strip()
            else: # if it's not a string (e.g. dict from tool call that wasn't parsed into answer/context)
                final_agent_response_text = str(final_agent_response_text)

            # Typing indicator for realism
            log_and_print("[bold cyan]Tutor is typing...[/bold cyan]", level=logging.DEBUG)
            time.sleep(0.7)

            # Try to parse as JSON for structured responses
            try:
                parsed = json.loads(final_agent_response_text)
                if isinstance(parsed, dict):
                    # If it's a remediation response, format it nicely
                    if parsed.get("status") == "remediation_syllabus_content":
                        log_and_print(Markdown(parsed.get("message", "No remediation content available")))
                    else:
                        # For other structured responses, show the message
                        log_and_print(Markdown(parsed.get("message", str(parsed))))
                else:
                    log_and_print(Markdown(final_agent_response_text))
            except json.JSONDecodeError:
                # If not JSON, display as markdown
                log_and_print(Markdown(final_agent_response_text))
        except Exception as e:
            final_agent_response_text = f"Error processing agent response: {e}"
            if debug:
                log_and_print(f"[bold red]Error in response extraction: {e}[/bold red]", level=logging.ERROR)

        # Store Interaction
        conversation_memory.add_interaction(
            student_id=current_student_id,
            user_query=question, # Store the original, non-augmented question
            retrieved_context=actual_rag_context_used_this_turn, # This needs to be correctly populated
            agent_response=final_agent_response_text # This is just the 'answer' part. For memory, we might want the full JSON if it helps reconstruct context later, or be more specific.
                                                     # For now, storing the text response.
        )

@app.command()
def start(
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode to see detailed information about the system\'s operation")
):
    """Start an interactive session with the ZIMSEC Tutoring System"""
    log_file_path = "session.log"
    log_level = logging.DEBUG if debug else logging.INFO

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to avoid duplication if re-running in same session/notebook
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close() # Close handlers before removing

    # File handler (always writes to session.log)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level) 
    root_logger.addHandler(file_handler)

    # Console stream handler (for Python's logging, respects debug flag for console verbosity)
    # Rich console output is handled by log_and_print directly for its own messages.
    # This handler is for other logging messages from libraries or our own code.
    if debug: # Only add stream handler for verbose console output if debug is true
        stream_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s') # Simpler for console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(stream_formatter)
        stream_handler.setLevel(logging.DEBUG) # Stream handler for DEBUG level
        root_logger.addHandler(stream_handler)
    
    logging.info("Logging initialized. Session log will be written to session.log")
    if debug:
        logging.debug("Debug mode enabled. Detailed logs will be written.")


    if not os.environ.get("OPENAI_API_KEY"):
        log_and_print("[bold red]Error: OPENAI_API_KEY environment variable not set.[/bold red]", level=logging.ERROR)
        sys.exit(1)

    asyncio.run(interactive_session(debug))

if __name__ == "__main__":
    app() 