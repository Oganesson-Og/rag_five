#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the MathProcessor class.
"""

import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger('MathProcessorTest')

# Add the project root to the Python path
def get_project_base_directory():
    """Get the base directory of the project."""
    return os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, get_project_base_directory())

# Import the MathProcessor
from src.document_processing.processors.math_processor import MathProcessor, MathProcessorConfig

def test_has_math_content():
    """Test the has_math_content method with various types of mathematical content."""
    # Initialize the MathProcessor
    config = MathProcessorConfig()
    processor = MathProcessor(config)
    
    # Test cases with mathematical content
    math_test_cases = [
        # LaTeX expressions
        "$x^2 + y^2 = z^2$",
        "The Pythagorean theorem is $a^2 + b^2 = c^2$",
        "\\begin{equation}E = mc^2\\end{equation}",
        
        # Mathematical symbols and operators
        "The value of π is approximately 3.14159",
        "If x > 5, then the function is positive",
        "The sum of the series is ∑i=1 to n",
        
        # Equation patterns
        "Solve for x: 2x + 3 = 7",
        "The quadratic formula is x = (-b ± √(b² - 4ac)) / 2a",
        "The derivative of sin(x) is cos(x)",
        
        # Common mathematical terms
        "In calculus, we study limits and derivatives",
        "The theorem states that every polynomial has a root",
        "The matrix has eigenvalues λ₁ and λ₂",
        
        # Numerical patterns
        "2 + 3 = 5",
        "If x = 5, then y = 10",
        "The function f(x) = x² is always non-negative"
    ]
    
    # Test cases without mathematical content
    non_math_test_cases = [
        "This is a plain text without any mathematical content.",
        "The quick brown fox jumps over the lazy dog.",
        "Today is a sunny day with clear skies.",
        "She went to the store to buy some groceries.",
        "The movie was entertaining and had a great plot."
    ]
    
    # Test math content detection
    logger.info("Testing detection of mathematical content...")
    for i, test_case in enumerate(math_test_cases):
        result = processor.has_math_content(test_case)
        logger.info(f"Math Test Case {i+1}: {'✓' if result else '✗'} - {test_case[:50]}...")
        assert result, f"Failed to detect mathematical content in: {test_case}"
    
    # Test non-math content detection
    logger.info("\nTesting detection of non-mathematical content...")
    for i, test_case in enumerate(non_math_test_cases):
        result = processor.has_math_content(test_case)
        logger.info(f"Non-Math Test Case {i+1}: {'✗' if not result else '✓'} - {test_case[:50]}...")
        assert not result, f"Incorrectly detected mathematical content in: {test_case}"
    
    # Test edge cases
    logger.info("\nTesting edge cases...")
    edge_cases = [
        ("", False, "Empty string"),
        (None, False, "None value"),
        ("   ", False, "Whitespace only"),
        ("x = 5", True, "Simple variable assignment"),
        ("The temperature is 25°C", False, "Number with unit"),
        ("Chapter 5: Introduction", False, "Number in text"),
        ("f(x)", True, "Function notation"),
        ("sin(θ)", True, "Trigonometric function")
    ]
    
    for test_case, expected, description in edge_cases:
        try:
            result = processor.has_math_content(test_case)
            status = "✓" if result == expected else "✗"
            logger.info(f"Edge Case - {description}: {status} - {'Detected' if result else 'Not detected'}")
            assert result == expected, f"Failed edge case '{description}': expected {expected}, got {result}"
        except Exception as e:
            logger.error(f"Error processing edge case '{description}': {str(e)}")
    
    logger.info("\nAll tests completed successfully!")

def test_process_method():
    """Test the process method with and without mathematical content."""
    # Initialize the MathProcessor
    config = MathProcessorConfig()
    processor = MathProcessor(config)
    
    # Test with mathematical content
    math_text = """
    The quadratic formula is used to solve quadratic equations of the form ax² + bx + c = 0.
    The solution is given by:
    
    $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$
    
    For example, to solve 2x² + 5x - 3 = 0:
    a = 2, b = 5, c = -3
    
    Step 1: Calculate the discriminant: b² - 4ac = 5² - 4(2)(-3) = 25 + 24 = 49
    Step 2: Apply the formula: x = (-5 ± √49) / 4 = (-5 ± 7) / 4
    Step 3: Simplify: x₁ = (-5 + 7) / 4 = 2/4 = 0.5, x₂ = (-5 - 7) / 4 = -12/4 = -3
    
    Therefore, the solutions are x = 0.5 and x = -3.
    """
    
    # Test with non-mathematical content
    non_math_text = """
    The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reefs and 900 islands.
    It is located in the Coral Sea, off the coast of Queensland, Australia.
    The reef is home to thousands of species of marine life, including fish, turtles, and dolphins.
    It is a popular destination for tourists and is considered one of the seven natural wonders of the world.
    """
    
    logger.info("Testing process method with mathematical content...")
    math_result = processor.process(math_text)
    logger.info(f"Has math content: {math_result['has_math_content']}")
    logger.info(f"Number of equations: {math_result['total_equations']}")
    logger.info(f"Complexity: {math_result['complexity']}")
    
    logger.info("\nTesting process method with non-mathematical content...")
    non_math_result = processor.process(non_math_text)
    logger.info(f"Has math content: {non_math_result['has_math_content']}")
    logger.info(f"Number of equations: {non_math_result['total_equations']}")
    
    assert math_result['has_math_content'], "Failed to detect math content in mathematical text"
    assert not non_math_result['has_math_content'], "Incorrectly detected math content in non-mathematical text"
    
    logger.info("\nProcess method tests completed successfully!")

if __name__ == "__main__":
    logger.info("Starting MathProcessor tests...")
    test_has_math_content()
    test_process_method()
    logger.info("All tests completed successfully!") 