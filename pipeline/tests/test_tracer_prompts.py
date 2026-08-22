import pytest

from qcd.ground_truth.tracer_prompts import (
    NORMALIZATION_PROMPT,
    TRACER_PROMPT_VERSION,
    TRIVIAL_SCREENING_PROMPT,
    VERIFICATION_PROMPT,
    render_normalization_prompt,
    render_trivial_screening_prompt,
    render_verification_prompt,
)


def test_prompt_version_identifies_source_revision_and_tables():
    assert TRACER_PROMPT_VERSION == "arxiv-2605.24079v1-appendix-a-tables-7-9"


def test_normalization_prompt_retains_published_examples_and_output_cue():
    assert "convert an image wrapper" in NORMALIZATION_PROMPT
    assert "maximum value in record list" in NORMALIZATION_PROMPT
    assert NORMALIZATION_PROMPT.endswith("Rephrased Task Description\n[New description here]")


def test_verification_prompt_retains_all_labels_examples_and_output_contract():
    for heading in (
        "A. Functionally Identical",
        "B. Nearly Identical",
        "C. Shared Logic",
        "D. Unrelated or Different Domain",
    ):
        assert heading in VERIFICATION_PROMPT
    assert VERIFICATION_PROMPT.count("Example ") == 5
    assert VERIFICATION_PROMPT.endswith("Answer: [A, B, C, or D]")


def test_trivial_prompt_retains_all_published_litmus_tests():
    assert "Built-in mapping:" in TRIVIAL_SCREENING_PROMPT
    assert "Subroutine usage:" in TRIVIAL_SCREENING_PROMPT
    assert "Atomic simplicity:" in TRIVIAL_SCREENING_PROMPT
    assert "Decision: Yes | No" in TRIVIAL_SCREENING_PROMPT
    assert "Reasoning: (3–4 sentences" in TRIVIAL_SCREENING_PROMPT


def test_normalization_renderer_inserts_description_literally():
    description = "Return {description1}; keep $values and [brackets]."
    rendered = render_normalization_prompt(description)
    assert description in rendered
    assert "{original_description}" not in rendered


def test_verification_renderer_does_not_interpolate_placeholders_inside_input():
    first = "Literal {description2} must remain."
    second = "Literal {description1} must also remain."
    rendered = render_verification_prompt(first, second)
    assert f"Task A. {first}" in rendered
    assert f"Task B. {second}" in rendered


def test_trivial_renderer_inserts_multiline_task_without_modification():
    task = "Line one\nLine two with {task_description}"
    assert render_trivial_screening_prompt(task).endswith("Task\n" + task)


@pytest.mark.parametrize(
    ("renderer", "args"),
    [
        (render_normalization_prompt, (None,)),
        (render_verification_prompt, ("valid", 1)),
        (render_trivial_screening_prompt, ([],)),
    ],
)
def test_renderers_reject_non_string_descriptions(renderer, args):
    with pytest.raises(TypeError):
        renderer(*args)


def test_prompts_do_not_invent_system_roles_or_decoding_settings():
    combined = "\n".join((NORMALIZATION_PROMPT, VERIFICATION_PROMPT, TRIVIAL_SCREENING_PROMPT))
    for absent in ("system prompt", "temperature", "top_p", "max_tokens"):
        assert absent not in combined.casefold()
