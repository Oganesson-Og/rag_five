import re

def debug_patterns():
    text = "The movie was entertaining and had a great plot."
    print(f"Debugging patterns for text: '{text}'")
    
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
    
    print("\nChecking math symbols patterns:")
    for i, pattern in enumerate(math_symbols):
        match = re.search(pattern, text)
        if match:
            print(f"Pattern {i+1}: '{pattern}' - Match: '{match.group(0)}'")
    
    # Check for equation patterns
    equation_patterns = {
        'basic': r'(\d+[\+\-\*/]\d+[=]\d+)',
        'algebraic': r'([a-z]\s*=\s*[-+]?\d*\.?\d+)',
        'quadratic': r'([-+]?\d*x\^2\s*[-+]\s*\d*x\s*[-+]\s*\d+\s*=\s*0)',
        'trigonometric': r'(sin|cos|tan)[\s\(].*[\)]',
        'calculus': r'(\∫|\\lim|\\frac\{d\}\{dx\})',
    }
    
    print("\nChecking equation patterns:")
    for name, pattern in equation_patterns.items():
        match = re.search(pattern, text)
        if match:
            print(f"Pattern '{name}': '{pattern}' - Match: '{match.group(0)}'")
    
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
        r'\b(x-axis|y-axis|z-axis|coordinate|graph|plot|curve)\b'
    ]
    
    print("\nChecking math keywords patterns:")
    for i, pattern in enumerate(math_keywords):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            print(f"Pattern {i+1}: '{pattern}' - Match: '{match.group(0)}'")
    
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
    
    print("\nChecking numerical patterns:")
    for i, pattern in enumerate(numerical_patterns):
        match = re.search(pattern, text)
        if match:
            print(f"Pattern {i+1}: '{pattern}' - Match: '{match.group(0)}'")

if __name__ == "__main__":
    debug_patterns() 