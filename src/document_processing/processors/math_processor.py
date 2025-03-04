"""
Mathematical Content Processor
----------------------------

Handles processing and analysis of mathematical content within documents,
specifically tailored for high school mathematics.

Key Features:
- Equation extraction and parsing
- Mathematical symbol recognition
- Formula identification
- Topic classification
- Complexity analysis
- LaTeX conversion
- Step-by-step solution parsing
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import re
import sympy
from sympy.parsing.latex import parse_latex
import numpy as np
from enum import Enum

class MathTopic(Enum):
    """High school math topics."""
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    TRIGONOMETRY = "trigonometry"
    CALCULUS = "calculus"
    STATISTICS = "statistics"
    UNKNOWN = "unknown"

@dataclass
class MathProcessorConfig:
    """Math processor configuration."""
    complexity_threshold: float = 0.7
    extract_latex: bool = True
    parse_solutions: bool = True
    detect_topics: bool = True
    min_confidence: float = 0.5
    max_equations_per_chunk: int = 10

@dataclass
class Equation:
    """Represents a mathematical equation."""
    text: str
    latex: Optional[str] = None
    topic: Optional[MathTopic] = None
    complexity: float = 0.0
    variables: List[str] = None
    is_solved: bool = False
    solution_steps: List[str] = None

class MathProcessor:
    """Handles mathematical content processing for high school level content."""
    
    def __init__(self, config: MathProcessorConfig):
        self.config = config
        self.equation_patterns = {
            'basic': r'(\d+[\+\-\*/]\d+[=]\d+)',
            'algebraic': r'([a-z]\s*=\s*[-+]?\d*\.?\d+)',
            'quadratic': r'([-+]?\d*x\^2\s*[-+]\s*\d*x\s*[-+]\s*\d+\s*=\s*0)',
            'trigonometric': r'(sin|cos|tan)[\s\(].*[\)]',
            'calculus': r'(\∫|\\lim|\\frac\{d\}\{dx\})',
        }
        self._has_math_content = False
        
    def has_math_content(self, text: str) -> bool:
        """
        Check if the text contains mathematical content.
        
        This method uses multiple detection strategies to determine if the text
        contains mathematical elements, including:
        - LaTeX expressions
        - Mathematical symbols and operators
        - Equation patterns
        - Common mathematical terms and keywords
        
        Args:
            text: Input text to check for mathematical content
            
        Returns:
            bool: True if mathematical content is detected, False otherwise
        """
        if not text or len(text.strip()) == 0:
            return False
            
        # Check for LaTeX expressions
        latex_patterns = [
            r'\$.*?\$',                     # Inline LaTeX
            r'\$\$.*?\$\$',                 # Display LaTeX
            r'\\begin\{equation\}.*?\\end\{equation\}',  # Equation environment
            r'\\begin\{align\}.*?\\end\{align\}',        # Align environment
            r'\\begin\{math\}.*?\\end\{math\}',          # Math environment
            r'\\begin\{displaymath\}.*?\\end\{displaymath\}',  # Display math environment
            r'\\begin\{eqnarray\}.*?\\end\{eqnarray\}',  # Equation array environment
            r'\\begin\{multline\}.*?\\end\{multline\}',  # Multiline environment
            r'\\begin\{gather\}.*?\\end\{gather\}',      # Gather environment
            r'\\begin\{flalign\}.*?\\end\{flalign\}'     # Full-length align environment
        ]
        
        for pattern in latex_patterns:
            if re.search(pattern, text, re.DOTALL):
                return True
                
        # Check for mathematical symbols and operators
        math_symbols = [
            r'[=<>≤≥≈≠∈∉⊂⊃∪∩]',            # Comparison and set operators
            r'[+\-*/÷×±]',                  # Basic arithmetic operators
            r'[∑∏∫∂∇∆√∛∜]',                # Advanced mathematical symbols
            r'[πΠθΘαβγΓλΛφΦ]',              # Greek letters commonly used in math
            r'[∞∀∃∄∴∵]',                   # Logical and infinity symbols
            r'[\^_]',                       # Superscript and subscript indicators
            r'\\frac\{.*?\}\{.*?\}',        # Fractions
            r'\\sqrt\{.*?\}'                # Square roots
        ]
        
        for pattern in math_symbols:
            if re.search(pattern, text):
                return True
                
        # Check for equation patterns
        for pattern in self.equation_patterns.values():
            if re.search(pattern, text):
                return True
                
        # Check for common mathematical terms and keywords
        math_keywords = [
            r'\b(equation|formula|theorem|lemma|proof|solve|calculate|compute)\b',
            r'\b(algebra|geometry|calculus|trigonometry|statistics|probability)\b',
            r'\b(function|variable|constant|coefficient|exponent|logarithm)\b',
            r'\b(matrix|vector|scalar|tensor|determinant|eigenvalue)\b',
            r'\b(triangle|circle|square|rectangle|polygon|angle|degree|radian)\b',
            r'\b(mean|median|mode|variance|standard deviation|distribution)\b',
            r'\b(derivative|integral|limit|differential|series|sequence)\b',
            r'\b(sin|cos|tan|arcsin|arccos|arctan|sinh|cosh|tanh)\b',
            r'\b(polynomial|quadratic|cubic|linear|exponential|logarithmic)\b',
            r'\b(x-axis|y-axis|z-axis|coordinate|graph)\b',
            r'\b(plot\s+(?:the\s+)?(?:graph|function|equation|data|points|curve))\b',
            r'\b(curve\s+(?:the\s+)?(?:graph|function|equation|line))\b'
        ]
        
        for pattern in math_keywords:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        # Check for numerical patterns that suggest mathematical content
        numerical_patterns = [
            r'\d+\s*[+\-*/÷×]\s*\d+',       # Basic arithmetic: 2 + 3
            r'\d+\s*=\s*\d+',               # Equality: 2 = 2
            r'[a-zA-Z]\s*[+\-*/÷×]\s*[a-zA-Z]',  # Variable operations: a + b
            r'[a-zA-Z]\s*=\s*\d+',          # Variable assignment: x = 5
            r'[a-zA-Z]\(\s*[a-zA-Z0-9,\s]+\s*\)',  # Function calls: f(x)
            r'\d+\s*[<>≤≥]\s*\d+',          # Inequalities: 5 > 3
            r'\(\s*\d+\s*[+\-*/÷×]\s*\d+\s*\)'  # Parenthesized expressions: (2 + 3)
        ]
        
        for pattern in numerical_patterns:
            if re.search(pattern, text):
                return True
                
        # If none of the above patterns matched, assume no mathematical content
        return False
        
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process mathematical content in text.
        
        Args:
            text: Input text containing mathematical content
            
        Returns:
            Dictionary containing processed mathematical information
        """
        try:
            # Check if the text contains mathematical content
            self._has_math_content = self.has_math_content(text)
            
            # If no mathematical content, return early with empty results
            if not self._has_math_content:
                return {
                    "equations": [],
                    "complexity": 0.0,
                    "topic_distribution": {},
                    "has_solutions": False,
                    "total_equations": 0,
                    "has_math_content": False
                }
            
            # Extract equations
            equations = self._extract_equations(text)
            
            # Classify topics
            if self.config.detect_topics:
                self._classify_topics(equations)
            
            # Parse solutions if present
            if self.config.parse_solutions:
                self._parse_solutions(equations, text)
            
            # Calculate overall complexity
            complexity = self._calculate_complexity(equations)
            
            return {
                "equations": [self._equation_to_dict(eq) for eq in equations],
                "complexity": complexity,
                "topic_distribution": self._get_topic_distribution(equations),
                "has_solutions": any(eq.is_solved for eq in equations),
                "total_equations": len(equations),
                "has_math_content": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "equations": [],
                "complexity": 0.0,
                "has_math_content": False
            }

    def _extract_equations(self, text: str) -> List[Equation]:
        """Extract equations from text."""
        equations = []
        
        # Extract LaTeX equations
        latex_pattern = r'\$(.*?)\$'
        latex_matches = re.finditer(latex_pattern, text)
        for match in latex_matches:
            latex = match.group(1)
            try:
                # Parse LaTeX to check validity
                parsed = parse_latex(latex)
                equations.append(Equation(
                    text=str(parsed),
                    latex=latex,
                    variables=list(parsed.free_symbols)
                ))
            except:
                continue
        
        # Extract other mathematical patterns
        for pattern_name, pattern in self.equation_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                eq_text = match.group(0)
                if not any(eq.text == eq_text for eq in equations):  # Avoid duplicates
                    equations.append(Equation(
                        text=eq_text,
                        variables=self._extract_variables(eq_text)
                    ))
                    
        return equations[:self.config.max_equations_per_chunk]

    def _classify_topics(self, equations: List[Equation]):
        """Classify equations by topic."""
        for eq in equations:
            # Check for topic indicators
            if any(trig in eq.text.lower() for trig in ['sin', 'cos', 'tan']):
                eq.topic = MathTopic.TRIGONOMETRY
            elif 'lim' in eq.text or 'dx' in eq.text or '∫' in eq.text:
                eq.topic = MathTopic.CALCULUS
            elif '^2' in eq.text or 'sqrt' in eq.text:
                eq.topic = MathTopic.ALGEBRA
            elif any(geo in eq.text.lower() for geo in ['triangle', 'circle', 'angle']):
                eq.topic = MathTopic.GEOMETRY
            elif any(stat in eq.text.lower() for stat in ['mean', 'median', 'variance']):
                eq.topic = MathTopic.STATISTICS
            else:
                eq.topic = MathTopic.UNKNOWN

    def _parse_solutions(self, equations: List[Equation], text: str):
        """Parse solution steps if present."""
        solution_pattern = r'(?:solution|steps?|answer):\s*((?:[1-9]\..*(?:\n|$))+)'
        for eq in equations:
            matches = re.finditer(solution_pattern, text, re.IGNORECASE)
            for match in matches:
                steps = match.group(1).split('\n')
                steps = [s.strip() for s in steps if s.strip()]
                if steps:
                    eq.is_solved = True
                    eq.solution_steps = steps

    def _calculate_complexity(self, equations: List[Equation]) -> float:
        """Calculate overall mathematical complexity."""
        if not equations:
            return 0.0
            
        complexities = []
        for eq in equations:
            # Base complexity
            complexity = 0.5
            
            # Adjust based on variables
            if eq.variables:
                complexity += min(len(eq.variables) * 0.1, 0.3)
            
            # Adjust based on topic
            if eq.topic in [MathTopic.CALCULUS, MathTopic.TRIGONOMETRY]:
                complexity += 0.2
            
            # Adjust based on equation length
            complexity += min(len(eq.text) / 100, 0.2)
            
            complexities.append(complexity)
            
        return min(1.0, np.mean(complexities))

    def _extract_variables(self, equation: str) -> List[str]:
        """Extract variables from equation text."""
        variables = re.findall(r'[a-zA-Z]', equation)
        return list(set(variables))

    def _get_topic_distribution(self, equations: List[Equation]) -> Dict[str, float]:
        """Get distribution of mathematical topics."""
        if not equations:
            return {}
            
        topic_counts = {}
        for eq in equations:
            if eq.topic:
                topic_counts[eq.topic.value] = topic_counts.get(eq.topic.value, 0) + 1
                
        total = len(equations)
        return {topic: count/total for topic, count in topic_counts.items()}

    def _equation_to_dict(self, eq: Equation) -> Dict[str, Any]:
        """Convert Equation object to dictionary."""
        return {
            "text": eq.text,
            "latex": eq.latex,
            "topic": eq.topic.value if eq.topic else None,
            "complexity": eq.complexity,
            "variables": eq.variables,
            "is_solved": eq.is_solved,
            "solution_steps": eq.solution_steps
        } 