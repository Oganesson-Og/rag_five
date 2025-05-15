"""
ZIMSEC Tutoring System - Agents Package
---------------------------------------

This package initializes and exports all the custom AI agent classes used within
the ZIMSEC Tutoring System. It serves as a central point for accessing the
different agent implementations.

By importing agents here and defining `__all__`, other parts of the system can
conveniently import specific agents directly from the `zimsec_tutor_agent_system.agents`
package.

Key Agents Exported:
- `CurriculumAlignmentAgent`: Aligns student queries with the syllabus.
- `OrchestratorAgent`: Routes queries to appropriate specialist agents.
- `ConceptTutorAgent`: Provides explanations and tutoring on specific concepts.
- `DiagnosticRemediationAgent`: Identifies learning gaps and suggests remediation.
- `AssessmentRevisionAgent`: Helps students with assessment and revision tasks.
- `ProjectsMentorAgent`: Assists students with CALA projects.
- `ContentGenerationAgent`: Generates educational content (e.g., examples, practice questions).
- `AnalyticsProgressAgent`: Tracks student progress and provides analytics.
- `StudentInterfaceAgent`: Manages the direct interaction with the student/user proxy.

Author: Keith Satuku
Version: 1.0.0
Created: 2024
License: MIT
"""

# This __init__.py file makes Python treat the 'agents' directory as a package.
# We will add our custom agent definitions here or import them into this file.

from .curriculum_alignment_agent import CurriculumAlignmentAgent
from .orchestrator_agent import OrchestratorAgent
from .concept_tutor_agent import ConceptTutorAgent
from .diagnostic_remediation_agent import DiagnosticRemediationAgent
from .assessment_revision_agent import AssessmentRevisionAgent
# Import new agents
from .projects_mentor_agent import ProjectsMentorAgent
from .content_generation_agent import ContentGenerationAgent
from .analytics_progress_agent import AnalyticsProgressAgent
from .student_interface_agent import StudentInterfaceAgent

__all__ = [
    "CurriculumAlignmentAgent",
    "OrchestratorAgent",
    "ConceptTutorAgent",
    "DiagnosticRemediationAgent",
    "AssessmentRevisionAgent",
    "ProjectsMentorAgent",
    "ContentGenerationAgent",
    "AnalyticsProgressAgent",
    "StudentInterfaceAgent",
] 