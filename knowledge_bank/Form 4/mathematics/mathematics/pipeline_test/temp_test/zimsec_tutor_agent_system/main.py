import asyncio
import autogen

import sys # Add this
import os # Add this

# Add the parent directory (temp_test) to sys.path to allow absolute imports
# This makes 'zimsec_tutor_agent_system' and 'rag_oo_pipeline' importable as top-level packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now use absolute imports based on the new sys.path
from zimsec_tutor_agent_system.agents.curriculum_alignment_agent import CurriculumAlignmentAgent
from zimsec_tutor_agent_system.agents.orchestrator_agent import OrchestratorAgent
from zimsec_tutor_agent_system.agents.concept_tutor_agent import ConceptTutorAgent

# Configuration for LLM
# This example assumes you might be using a local LLM (like Ollama)
# that doesn't require an explicit API key in the config_list.
# If you're using OpenAI or another service, you'll need to configure this
# with your API key and potentially other parameters.
# For Ollama, ensure your Ollama server is running and the model is pulled.
config_list = [
    {
        "model": "phi4:latest",  # Replace with your desired model if different
        "base_url": "http://localhost:11434/v1", # Standard Ollama API endpoint
        "api_type": "ollama", # "open_ai" if using OpenAI compatible endpoint for Ollama
        "api_key": "ollama", # Placeholder, not strictly needed for local Ollama
    }
]

# More robust configuration for phi3 / ollama
# config_list = [
# {
# 'model': 'phi3', # or the specific ollama model name you have
# 'base_url': 'http://localhost:11434/v1',
# 'api_type': 'open_ai', # Use 'open_ai' for ollama models through LiteLLM or a compatible server
# 'api_key': 'NULL' # Required by autogen but not used by local ollama
# }
# ]

# 1. Instantiate CurriculumAlignmentAgent
curriculum_agent = CurriculumAlignmentAgent(
    name="CurriculumAlignmentAgent",
    # This agent uses its registered reply function with mock logic, 
    # so it doesn't strictly need an LLM for its core response generation for now.
    # However, Autogen might still expect llm_config for other internal things or if we add LLM-based replies later.
    llm_config={"config_list": config_list} 
)

concept_tutor = ConceptTutorAgent(
    name="ConceptTutorAgent",
    llm_config={"config_list": config_list}
)

# 2. Instantiate OrchestratorAgent
# It needs the curriculum_agent to delegate alignment checks to.
orchestrator_agent = OrchestratorAgent(
    name="OrchestratorAgent",
    llm_config={"config_list": config_list},
    curriculum_alignment_agent=curriculum_agent,
    concept_tutor_agent=concept_tutor
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

    initial_query = "Tell me about the median from a grouped frequency table."
    # initial_query = "What is the difference between speed and velocity?"
    # initial_query = "What is the capital of France?"

    await user_proxy.a_initiate_chat(
        recipient=orchestrator_agent,
        message=initial_query,
        max_turns=2 # Ensure max_turns is 2 to prevent auto-reply in TERMINATE mode
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