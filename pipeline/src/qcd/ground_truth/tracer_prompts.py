"""Verbatim prompt templates from TRACER Appendix A, Tables 7--9.

The renderers only insert task descriptions.  They intentionally define no
system prompt, sampling parameters, or model-specific message envelope because
those are not part of the published prompt tables.
"""

from __future__ import annotations

import re


TRACER_PROMPT_VERSION = "arxiv-2605.24079v1-appendix-a-tables-7-9"


NORMALIZATION_PROMPT = """Instruction
Carefully read the programming task and the examples provided. Then rephrase the
original task description into clean and concise ones. Make sure the rephrased
task description follow the style and length of rephrased ones provided in the
examples. Directly return the rephrased task description.

Example 1
Original Task Description
You are tasked with implementing a function to convert an image wrapper to a
OpenGL texture. The image wrapper is a data structure that holds image data, and
the OpenGL texture is a representation of the image suitable for rendering in an
OpenGL environment.
Rephrased Task Description
Implement a method that converts an image wrapper containing image data into a
texture suitable for rendering in an OpenGL environment. The method should handle
data formatting and texture creation within a properly initialized OpenGL context,
ensuring correctness and efficiency.

Example 2
Original Task Description
Write a function to find the maximum value in record list as tuple attribute in
the given tuple list.
Rephrased Task Description
Implement a method that processes a list of tuples and returns the maximum value
found among a specific attribute within each tuple. The method should correctly
extract and compare values to determine the highest one.

Task
Original Task Description
{original_description}
Rephrased Task Description
[New description here]"""


VERIFICATION_PROMPT = """Instruction
1. You will see two tasks: Task A and Task B.
2. Read both carefully, noting their goals, inputs/outputs, and logic.
3. Choose the single most accurate relationship from the categories below.

Relationship Categories
A. Functionally Identical
Choose this if the tasks are perfect duplicates. They accomplish the exact same
goal, take the same kinds of input, and produce the same kinds of output. They are
essentially two descriptions of the very same problem.
Litmus Test: Could the solution for one task solve the other with zero changes?
If yes, choose A. Otherwise, do NOT choose A.

B. Nearly Identical (Variation of the Same Problem)
Choose this if the tasks solve the same fundamental problem, but differ only in
minor surface details, but share all the same core logic. They solve the same
fundamental problem, but with minor differences in constraints, data types, or
input/output formats.
Litmus Test: If the tasks are not perfectly identical (A fails), but the
**core logic is identical**, choose B. If the core logic differs, do NOT choose B.

C. Shared Logic (Different Problems, Same Algorithm)
Choose this if the tasks solve different problems using the same algorithmic
method. The tasks have different goals and may come from unrelated domains, but
they are solved using the same core algorithm or logical procedure.
Litmus Test: If neither A nor B applies, but the algorithmic approach is the same,
choose C. If the algorithm differs, do NOT choose C.

D. Unrelated or Different Domain
Choose this if the tasks do not share the same algorithmic logic. This includes
two cases:
The tasks are from the same general domain (e.g., both deal with arrays or graphs)
but require different algorithms or solution methods.
The tasks are completely unrelated –- they have no meaningful conceptual,
logical, or domain connection.
Litmus Test: If none of A, B, or C applies, choose D.

Examples
Example 1
Task A:
Determine if a given string is a palindrome, returning True if it reads the same
backward as forward.
Task B:
Implement a method in Ruby that determines whether a given string is a palindrome.
Answer: A

Example 2
Task A:
Generate a space-delimited string of numbers starting from 0 up to n inclusive.
Task B:
Implement a C++ function to print the numbers from 0 to n in ascending order.
Answer: B

Example 3
Task A:
Given an array of integers and a positive integer k, return a sorted list of the k
largest numbers in the array.
Task B:
Implement a function to identify the two largest numbers in an array and return
them in descending order.
Answer: C

Example 4
Task A:
Determine if any two numbers in the given list are closer to each other than a
specified threshold.
Task B:
Given a sorted integer array and two integers k and x, return the k closest
integers to x, sorted in ascending order. An integer is considered closer to x if
it has a smaller absolute difference, or the same difference but is smaller in
value.
Answer: D

Example 5
Task A:
Determine if any two numbers in the given list are closer to each other than a
specified threshold.
Task B:
Given an integer array nums, count the elements that have both a strictly smaller
and a strictly greater element in the array.
Answer: D

Input Tasks
Task A. {description1}
Task B. {description2}

Output Requirements
Format your answer exactly as follows:
Answer: [A, B, C, or D]"""


TRIVIAL_SCREENING_PROMPT = """Instruction
You will be shown one task description. Your job is to assess whether it
describes a basic helper function.

Definition
A basic helper function is:
1. Primitive and atomic –- performs a single, irreducible operation.
2. Scalar/boolean output –- returns only a simple scalar or trivial boolean (not
a composite structure).
3. Built-in equivalent –- typically maps to a single built-in or standard library
function (e.g., abs(x), len(list), max(array)).
4. Subroutine nature –- commonly used as a small sub-step inside larger
algorithms.

Litmus Tests (all must be satisfied for “Yes”)
- Built-in mapping: Does it directly correspond to a built-in/standard library
call?
- Subroutine usage: Is it normally a utility step within larger problems?
- Atomic simplicity: Does it avoid extra selection, indexing, or multi-step
logic?

Decision Rule
- Yes: All three tests pass.
- No: Any test fails.

Output Requirements
Format your answer exactly as follows:
Decision: Yes | No
Reasoning: (3–4 sentences explaining which tests pass or fail, focusing on atomic
simplicity, built-in mapping, and subroutine usage.)

Task
{task_description}"""


def _render(template: str, placeholder_values: tuple[tuple[str, str], ...]) -> str:
    values: dict[str, str] = {}
    for placeholder, value in placeholder_values:
        if not isinstance(value, str):
            raise TypeError(f"{placeholder[1:-1]} must be a string")
        values[placeholder] = value
    pattern = re.compile("|".join(re.escape(placeholder) for placeholder in values))
    # A single regex pass ensures placeholder-looking text supplied by a user
    # is never interpreted as another template field.
    return pattern.sub(lambda match: values[match.group(0)], template)


def render_normalization_prompt(original_description: str) -> str:
    return _render(NORMALIZATION_PROMPT, (("{original_description}", original_description),))


def render_verification_prompt(description1: str, description2: str) -> str:
    return _render(
        VERIFICATION_PROMPT,
        (("{description1}", description1), ("{description2}", description2)),
    )


def render_trivial_screening_prompt(task_description: str) -> str:
    return _render(TRIVIAL_SCREENING_PROMPT, (("{task_description}", task_description),))
