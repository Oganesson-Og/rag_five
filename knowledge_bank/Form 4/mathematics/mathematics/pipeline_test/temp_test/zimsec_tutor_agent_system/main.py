import asyncio
import autogen
import json

import sys # Add this
import os # Add this

# Add the parent directory (temp_test) to sys.path to allow absolute imports
# This makes 'zimsec_tutor_agent_system' and 'rag_oo_pipeline' importable as top-level packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now use absolute imports based on the new sys.path
from zimsec_tutor_agent_system.agents.curriculum_alignment_agent import CurriculumAlignmentAgent
from zimsec_tutor_agent_system.agents.orchestrator_agent import OrchestratorAgent
from zimsec_tutor_agent_system.agents.concept_tutor_agent import ConceptTutorAgent
from zimsec_tutor_agent_system.agents.diagnostic_remediation_agent import DiagnosticRemediationAgent
from zimsec_tutor_agent_system.agents.assessment_revision_agent import AssessmentRevisionAgent
from zimsec_tutor_agent_system.agents.projects_mentor_agent import ProjectsMentorAgent
from zimsec_tutor_agent_system.agents.content_generation_agent import ContentGenerationAgent
from zimsec_tutor_agent_system.agents.analytics_progress_agent import AnalyticsProgressAgent

# Shared config_list for Ollama models accessed via OpenAI API type
# Using qwen3:14b as per user's last selection. Removed timeout for now to pass Pydantic.
config_list_openai_ollama = [
    {
        "model": "qwen3:14b",
        "base_url": "http://localhost:11434/v1",
        "api_type": "openai",
        "api_key": "ollama" # Placeholder, not strictly needed for local Ollama
    }
]

# More robust configuration for phi3 / ollama
# config_list = [
# {
# 'model': 'phi3', # or the specific ollama model name you have
# 'base_url': 'http://localhost:11434/v1',
# 'api_type': 'openai', # Use 'openai' for ollama models through LiteLLM or a compatible server
# 'api_key': 'NULL' # Required by autogen but not used by local ollama
# }
# ]

# Define the schema for the project_checklist tool
PROJECT_CHECKLIST_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "project_checklist",
        "description": "Provides a checklist or guidance for a given CALA project milestone. If the milestone is 'plan', returns detailed Markdown for Stage 1. Otherwise, returns JSON for other milestones.",
        "parameters": {
            "type": "object",
            "properties": {
                "milestone": {
                    "type": "string",
                    "description": "The project milestone (e.g., 'plan', 'investigation', 'solution_dev', 'presentation', 'evaluation').",
                },
                "draft_snippet": {
                    "type": "string",
                    "description": "A snippet of the student's current draft for context. Can be the original query if no specific draft text is available.",
                },
            },
            "required": ["milestone"], 
        },
    },
}

# 1. Instantiate CurriculumAlignmentAgent
curriculum_agent = CurriculumAlignmentAgent(
    name="CurriculumAlignmentAgent",
    # CurriculumAlignmentAgent might not need an LLM if its replies are fully rule-based/RAG-based
    # but Autogen expects llm_config. Using the shared one for consistency for now.
    llm_config={"config_list": config_list_openai_ollama}
)

concept_tutor = ConceptTutorAgent(
    name="ConceptTutorAgent",
    llm_config={"config_list": config_list_openai_ollama}
)

# Instantiate the new agents
diagnostic_agent = DiagnosticRemediationAgent(
    name="DiagnosticRemediationAgent",
    llm_config={"config_list": config_list_openai_ollama}
)

assessment_agent = AssessmentRevisionAgent(
    name="AssessmentRevisionAgent",
    llm_config={"config_list": config_list_openai_ollama}
)

# Instantiate the remaining agents
projects_mentor_agent = ProjectsMentorAgent(
    name="ProjectsMentorAgent",
    llm_config={
        "config_list": config_list_openai_ollama,
        "tools": [PROJECT_CHECKLIST_TOOL_SCHEMA]
        # Removed "timeout": 60 from here to avoid Pydantic error
    }
)

content_generation_agent = ContentGenerationAgent(
    name="ContentGenerationAgent",
    llm_config={"config_list": config_list_openai_ollama}
)

analytics_progress_agent = AnalyticsProgressAgent(
    name="AnalyticsProgressAgent",
    llm_config={"config_list": config_list_openai_ollama}
)

# 2. Instantiate OrchestratorAgent
# It needs the curriculum_agent to delegate alignment checks to.
orchestrator_agent = OrchestratorAgent(
    name="OrchestratorAgent",
    llm_config={"config_list": config_list_openai_ollama},
    curriculum_alignment_agent=curriculum_agent,
    concept_tutor_agent=concept_tutor,
    diagnostic_agent=diagnostic_agent,
    assessment_agent=assessment_agent,
    projects_mentor_agent=projects_mentor_agent,
    content_generation_agent=content_generation_agent,
    analytics_progress_agent=analytics_progress_agent
)

# 3. Instantiate UserProxyAgent (represents the human learner)
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="TERMINATE",  # Keep TERMINATE for controlled test
    is_termination_msg=lambda x: x.get("content", "").strip().lower() == "exit", # User can type 'exit' to end
    code_execution_config=False,
)

async def main_chat():
    print("Starting chat with Orchestrator Agent...")
    print("UserProxy will send an initial message.")
    print("(Orchestrator will route to specialists as needed)")

    # initial_query = "Tell me about the median from a grouped frequency table."
    # initial_query = "Give me some practice questions on Statistics"
    # initial_query = "Check my answer for the statistics problem: I think the median is 45.3"
    initial_query = "Help me with my CALA project plan." # Changed for testing Projects Mentor Agent
    initial_form_level = "Form 4" # Define the form level
    # initial_query = "What is the difference between speed and velocity?"

    # Structure the initial message as a JSON string
    initial_message_structured = json.dumps({
        "user_query": initial_query,
        "form_level": initial_form_level
    })

    await user_proxy.a_initiate_chat(
        recipient=orchestrator_agent,
        message=initial_message_structured, # Send the JSON string
        max_turns=1 # Ensure max_turns is 1 to prevent auto-reply in TERMINATE mode
    )
    print("Chat ended.")

# Start a chat from UserProxy to OrchestratorAgent
if __name__ == "__main__":
    asyncio.run(main_chat())

    # Example of a task that might require code execution
    # user_proxy.initiate_chat(
    #     assistant,
    #     message="Plot a sine wave from 0 to 2*pi and save it to a file named sine_wave.png.",
    # ) 