"""Real-network, GPU-free tests for qcd.real_run — the shared core behind
scripts/run_main.py. Confirms the parts of the
real (non-mock) pipeline that don't need a GPU: item loading/capping,
candidate-code assembly, and that `run()` fails at exactly the expected
place (model loading) rather than somewhere earlier due to a wiring bug —
leaving items.parquet and manifest.json already written when it does.
"""

import dataclasses
import datetime as dt
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from qcd.config import ModelSpec, Quant
from qcd.data.schema import Dataset, Item
from qcd.models.registry import QWEN2_5_7B
from qcd.real_run import RealRunConfig, _assemble_candidate_code, _generation_prompt, load_all_items, run


_TEST_QWEN = dataclasses.replace(
    QWEN2_5_7B, primary_first_post_boundary="2023-11-01"
)


def _small_config(tmp_path, **overrides) -> RealRunConfig:
    defaults = dict(
        models=(_TEST_QWEN,),
        quant_levels=(Quant.BF16,),
        output_dir=tmp_path,
        lcb_cutoff_boundary=dt.datetime(2023, 12, 1),
        lcb_release_version="release_v1",
        item_limit_per_condition=2,
    )
    defaults.update(overrides)
    return RealRunConfig(**defaults)


def test_load_all_items_respects_per_condition_cap(tmp_path):
    config = _small_config(tmp_path, item_limit_per_condition=2)
    items = load_all_items(config)

    counts: dict[Dataset, int] = {}
    for item in items:
        counts[item.dataset] = counts.get(item.dataset, 0) + 1

    for dataset, count in counts.items():
        assert count <= 2, f"{dataset} had {count} items, cap was 2"
    # All four conditions should be represented at this small scale.
    assert set(counts) == {Dataset.LCB_PRE, Dataset.LCB_POST, Dataset.HUMANEVAL, Dataset.MBPPPLUS}


def test_load_all_items_can_exclude_evalplus_datasets(tmp_path):
    config = _small_config(tmp_path, item_limit_per_condition=2, include_humaneval=False, include_mbppplus=False)
    items = load_all_items(config)
    datasets = {item.dataset for item in items}
    assert Dataset.HUMANEVAL not in datasets
    assert Dataset.MBPPPLUS not in datasets


# Real evalplus prompts (evalplus.data.get_human_eval_plus()) always end
# with a *closed* signature+docstring stub, never a dangling open indent —
# an unclosed prompt like "def f():\n    " would make prompt+completion
# concatenation parse as a (broken) nested function rather than two
# sibling top-level definitions, an artifact of the test fixture, not a
# real prompt shape.
_EVALPLUS_STUB_PROMPT = 'def f():\n    """docstring"""\n'


def _evalplus_item(entry_point: str, prompt: str = _EVALPLUS_STUB_PROMPT) -> Item:
    return Item(
        item_id="HumanEval/0", dataset=Dataset.HUMANEVAL, prompt=prompt,
        metadata={"evalplus_problem": {"entry_point": entry_point}},
    )


def test_assemble_candidate_code_prepends_prompt_for_evalplus_raw_continuation():
    # A raw, un-fenced continuation (no chat wrapping) should still work —
    # sanitize()'s AST extraction accepts this shape too, not just chat
    # output. The completion supplies its own leading indent, matching a
    # real raw-completion model continuing the prompt's closed docstring
    # stub as a fresh body line (not appended mid-line).
    item = _evalplus_item("f")
    result = _assemble_candidate_code(item, "    return 1\n")
    assert "return 1" in result


def test_assemble_candidate_code_extracts_function_from_chat_style_evalplus_completion():
    # Real shape confirmed on Qwen2.5-7B-Instruct output (2026-08-15 smoke
    # test): prose explanation + a fenced code block, not a raw continuation.
    item = _evalplus_item("f")
    completion = (
        "Sure! Here's the completed function:\n\n"
        "```python\ndef f():\n    return 1\n```\n\n"
        "This function takes no arguments and returns 1."
    )
    result = _assemble_candidate_code(item, completion)
    assert "def f():" in result
    assert "return 1" in result
    assert "Sure!" not in result
    assert "This function" not in result


def test_assemble_candidate_code_uses_completion_alone_for_lcb_raw():
    item = Item(item_id="q1", dataset=Dataset.LCB_PRE, prompt="Solve this problem.")
    result = _assemble_candidate_code(item, "print('hi')\n")
    assert "print('hi')" in result


def test_assemble_candidate_code_extracts_fenced_code_for_lcb_chat_completion():
    item = Item(item_id="q1", dataset=Dataset.LCB_PRE, prompt="Solve this problem.")
    completion = (
        "Looking at this problem, I need to read two integers and print their sum.\n\n"
        "```python\nimport sys\n\ndef main():\n    a, b = map(int, sys.stdin.readline().split())\n"
        "    print(a + b)\n\nmain()\n```\n\n"
        "This reads the input line, splits it, and prints the sum."
    )
    result = _assemble_candidate_code(item, completion)
    assert "def main():" in result
    assert "a + b" in result
    assert "Looking at this problem" not in result
    assert "This reads the input" not in result


def test_lcb_generation_prompt_adds_starter_without_changing_detector_prompt():
    original = "Return the smallest number."
    item = Item(
        item_id="functional", dataset=Dataset.LCB_POST, prompt=original,
        metadata={"starter_code": "class Solution:\n    def solve(self, n):\n        "},
    )
    rendered = _generation_prompt(item)
    assert original in rendered
    assert "class Solution" in rendered
    assert "Return only the complete code" in rendered
    assert item.prompt == original


def test_lcb_stdin_generation_prompt_requires_complete_program():
    item = Item(
        item_id="stdin", dataset=Dataset.LCB_POST, prompt="Add two integers.",
        metadata={"starter_code": ""},
    )
    rendered = _generation_prompt(item)
    assert "standard input" in rendered
    assert "standard output" in rendered
    assert "Return only the code" in rendered


def test_run_fails_at_model_loading_not_earlier(tmp_path, monkeypatch):
    # Do NOT actually call the real load_model() path here: on this profile
    # (no torch installed) it would try to download Qwen2.5-7B's tokenizer
    # over the network before failing for an unrelated reason (missing
    # torch), which is both slow/inappropriate for a test and not what this
    # test is checking. Stub load_model() to simulate "GPU path not ready
    # yet" directly, and confirm run()'s control flow reaches it (and only
    # it) after items/manifest are already written.
    import qcd.real_run as real_run_module

    def _stub_load_model(spec, quant, *, mock=False):
        raise NotImplementedError("real generation path — implement alongside scripts/run_smoke_test.py")

    monkeypatch.setattr(real_run_module, "load_model", _stub_load_model)

    config = _small_config(tmp_path, item_limit_per_condition=1)

    with pytest.raises(NotImplementedError, match="real generation path"):
        run(config)

    # Items and manifest should already be on disk — the failure happens
    # after those writes, not before.
    assert (tmp_path / "raw" / "items.parquet").exists()
    assert (tmp_path / "raw" / "model_item_labels.parquet").exists()
    assert (tmp_path / "manifest.json").exists()
    df = pd.read_parquet(tmp_path / "raw" / "items.parquet")
    assert len(df) > 0


def test_run_scores_fixed_prompt_and_keeps_completion_confidence(tmp_path, monkeypatch):
    import qcd.real_run as real_run_module

    item = Item(
        item_id="q1", dataset=Dataset.LCB_PRE, prompt="fixed prompt",
        metadata={"contest_date": "2023-06-01"},
    )

    class FakeModel:
        tokenizer = object()

        def __init__(self):
            self.prompt_calls = []

        def generate(self, item_id, prompt, *, temperature, sample_id):
            del item_id, prompt, sample_id
            return SimpleNamespace(
                text="print(1)", token_ids=[1, 2],
                token_logprobs=[-0.2, -0.3], is_greedy=temperature == 0.0,
            )

        def score_prompt_logprobs(self, item_id, prompt):
            self.prompt_calls.append((item_id, prompt))
            return [-1.0, -2.0, -3.0]

    fake_model = FakeModel()
    monkeypatch.setattr(real_run_module, "load_all_items", lambda config: [item])
    monkeypatch.setattr(
        real_run_module, "load_model", lambda spec, quant, mock=False: fake_model
    )
    monkeypatch.setattr(real_run_module, "_assemble_candidate_code", lambda item, text: text)
    monkeypatch.setattr(real_run_module, "partial_pass_rate", lambda item, code: 1.0)

    config = _small_config(
        tmp_path, n_cdd_samples=2, include_humaneval=False, include_mbppplus=False
    )
    run(config)

    assert fake_model.prompt_calls == [("q1", "fixed prompt")]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["extra"] == {}
    from qcd.io.manifest import config_hash
    assert manifest["config_hash"] == config_hash(manifest["config"])
    assert manifest["config"]["models"] == [QWEN2_5_7B.name]
    assert manifest["config"]["model_revisions"] == {
        QWEN2_5_7B.name: QWEN2_5_7B.revision,
    }
    assert manifest["config"]["baseline_dtype"] == "bfloat16"
    assert manifest["config"]["n_cdd_samples"] == 2
    assert manifest["config"]["model_primary_first_post_boundaries"] == {
        QWEN2_5_7B.name: "2023-11-01",
    }
    generations = pd.concat(
        [pd.read_parquet(path) for path in sorted((tmp_path / "raw").glob("generations*.parquet"))],
        ignore_index=True,
    )
    greedy = generations[generations["is_greedy"]].iloc[0]
    assert list(greedy["token_logprobs"]) == pytest.approx([-0.2, -0.3])
    assert list(greedy["prompt_token_logprobs"]) == pytest.approx([-1.0, -2.0, -3.0])

    scores = pd.concat(
        [pd.read_parquet(path) for path in sorted((tmp_path / "raw").glob("detector_scores*.parquet"))],
        ignore_index=True,
    )
    assert set(scores["detector"]) == {
        "cdd", "perplexity", "mink_prob",
        "completion_perplexity", "completion_mink_prob",
    }
