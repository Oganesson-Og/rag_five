import autogen
import json
# import random # No longer needed for the new tool logic
# import time # No longer needed for the new tool logic
from typing import List, Dict, Any, Optional, Tuple, Union

# Constants for the _mock_project_checklist_tool
EXAMPLE_PROJECT_TOPICS = [
    "1. Using Statistics to Analyze Daily Water Usage in Households: Investigate how much water is used per day and propose ways to reduce waste.",
    "2. Comparing Prices of Basic Goods in Local Markets Using Averages and Graphs: Analyze market price trends and determine where savings can be made.",
    "3. Using Geometry to Design Optimal Classroom Seating Arrangements: Apply area, perimeter, and layout planning for effective use of space.",
    "4. Measuring the Impact of Load Shedding on Study Time Using Data Charts: Collect and present data on electricity availability vs. student performance.",
    "5. Estimating Monthly Mobile Data Usage and Cost Efficiency Among Students: Use ratios, percentages, and graphs to analyze data plan usage."
]
STAGE_1_GUIDELINES = {
    "title": "Stage 1: Problem Identification (5 marks)",
    "criteria": [
        "1.1 Description of the problem, innovation or the identified gap (1 mark)",
        "1.2 Brief statement of intent (Linking to the problem, innovation or identified gap) - 2 marks",
        "1.3 Design / Project specification or parameters - at least 2 (2 marks)"
    ]
}

# Example project topics and Stage 1 guidelines are not needed here anymore
# as the tool logic will move to the Orchestrator.

class ProjectsMentorAgent(autogen.AssistantAgent):
    SYSTEM_PROMPT = """You are the ProjectsMentorAgent.
Your primary role is to guide students through O-Level Mathematics school-based projects (CALA).

You will receive a JSON message from the OrchestratorAgent. This message will look like:
{
  "task": "project_guidance",
  "milestone": "some_milestone", 
  "draft_snippet": "text from the user"
}

You have access to a function: `project_checklist` with parameters `milestone: str` and `draft_snippet: Optional[str]`.

Key Responsibilities & Tool Usage:
1.  When you receive a JSON message from the Orchestrator:
    a.  Examine the "milestone" field.
    b.  Examine the "draft_snippet" field.
2.  If the "milestone" field in the received JSON is "plan", you MUST call the `project_checklist` function.
    - For the `milestone` parameter of your function call, use the value "plan".
    - For the `draft_snippet` parameter of your function call, use the value from the "draft_snippet" field of the received JSON message.
3.  The `project_checklist` function, when called with `milestone="plan"`, will return detailed Markdown guidance for Stage 1 (this will come back to you in a subsequent `tool_response` message).
4.  After you receive this Markdown guidance in the `tool_response` message, your *next and final textual response for that turn* MUST BE the *exact, verbatim Markdown content* from that `tool_response`. Do NOT add any other text, conversation, or summarization around it.
5.  For any other "milestone" value (not "plan"), the `project_checklist` function will return JSON. You can then summarize or use that JSON to help the student after receiving it in the `tool_response`.

Example Interaction Flow (milestone="plan"):
1. Orchestrator sends you: `{"task": "project_guidance", "milestone": "plan", "draft_snippet": "User needs help with project plan."}`
2. You respond with a tool_calls message: Requesting `project_checklist(milestone="plan", draft_snippet="User needs help with project plan.")`
3. Orchestrator executes the tool and sends you a tool_response message containing Markdown.
4. You respond with only that Markdown content.
"""

    def __init__(self, name: str, llm_config: Dict, **kwargs):
        super().__init__(
            name,
            llm_config=llm_config,
            system_message=self.SYSTEM_PROMPT,
            **kwargs
        )
        # Register the mock tool
        self.register_function(
            function_map={
                "project_checklist": self._mock_project_checklist_tool
            }
        )

    # Tool method restored here
    def _mock_project_checklist_tool(self, milestone: str, draft_snippet: Optional[str] = None, current_stage_hint: Optional[int] = None) -> str:
        """
        MOCK TOOL: Provides a checklist or guidance for a given project milestone.
        If the milestone is "plan", this tool returns detailed Markdown text guiding the student through Stage 1: Problem Identification, based on official guidelines.
        Otherwise, it returns a JSON string with generic feedback for other milestones.
        Args:
            milestone (str): The project milestone. CRITICAL: If "plan", use Stage 1 guidance.
            draft_snippet (Optional[str]): A snippet of the student's current draft for context.
            current_stage_hint (Optional[int]): An optional hint for the current project stage (1-6).
        Returns:
            str: Markdown text for "plan" milestone; otherwise, a JSON string for other milestones.
        """
        print(f"[ProjectsMentorAgent Tool Method - _mock_project_checklist_tool] Called with milestone: {milestone}, Snippet: '{draft_snippet[:70] if draft_snippet else 'N/A'}...'")

        if milestone.lower() == "plan":
            guidance_parts = [
                f"Okay, let's focus on **{STAGE_1_GUIDELINES['title']}** for your project plan.",
                "This is a crucial first step. Here's what you need to cover based on the official guidelines:",
                ""
            ]
            for item in STAGE_1_GUIDELINES["criteria"]:
                description = item.split('(')[0].strip()
                marks = item[item.find('('):item.find(')')+1]
                guidance_parts.append(f"- **{description}** {marks}")

            guidance_parts.extend([
                "",
                "**To help you develop this section, consider these questions:**",
                "  - What specific real-world problem, observation, or gap in your community, school, or daily life can be explored using mathematical concepts from your syllabus?",
                "  - What exactly do you aim to achieve or find out through this project? (This is your statement of intent – it should clearly link to the problem you've identified).",
                "  - What are the key boundaries or specific aspects you will focus on? These are your project parameters or specifications. For example:",
                "    - Will your study be limited to a specific age group, a particular geographical location (e.g., your school, a local market), or a certain timeframe?",
                "    - What specific mathematical techniques or data will you be using?",
                "    - You need at least two clear specifications.",
                "",
                "**Need some inspiration for a topic? Here are a few general ideas that align with O-Level Mathematics projects:**"
            ])
            for i, topic in enumerate(EXAMPLE_PROJECT_TOPICS[:3]):
                 guidance_parts.append(f"  {i+1}. {topic.split(': ', 1)[1] if ': ' in topic else topic}")
            
            guidance_parts.extend([
                "",
                "Remember, your project plan (focused on Stage 1) is the foundation. Once you have a draft of your problem identification, statement of intent, and specifications, I can help you review it!"
            ])
            markdown_output = "\n".join(guidance_parts)
            print(f"[ProjectsMentorAgent Tool Method - _mock_project_checklist_tool] Returning Markdown for 'plan':\n{markdown_output[:400]}...")
            return markdown_output

        # Fallback for other milestones
        result = {
            "milestone": milestone,
            "status": "pending_review",
            "action_items": [
                f"Develop content for {milestone}.",
                "Ensure alignment with relevant project stage guidelines."
            ],
            "guidance_notes": "This is a general placeholder. More specific guidance will be developed for this milestone.",
            "contact_mentor_for_details": True
        }
        if draft_snippet:
            result["notes_on_snippet"] = "Snippet acknowledged. Detailed feedback requires specific stage context for milestones other than 'plan'."
        
        json_output = json.dumps(result)
        print(f"[ProjectsMentorAgent Tool Method - _mock_project_checklist_tool] Returning JSON for other milestones: {json_output}")
        return json_output

# Removed the overridden generate_reply method to use the base class's implementation 
# Removed _mock_project_checklist_tool method from here.
# Removed EXAMPLE_PROJECT_TOPICS and STAGE_1_GUIDELINES as they are tied to the tool method. 