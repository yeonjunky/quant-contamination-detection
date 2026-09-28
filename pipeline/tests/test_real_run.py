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
    assert manifest["study_phase"] == "main_study"
    # §4.3's resolved library defaults live in `extra`, not in `config`, so they
    # are recorded without entering `config_hash` (an environment change must
    # not read as a different run configuration).
    assert set(manifest["extra"]) == {
        "library_default_settings", "library_default_settings_unresolved"
    }
    from qcd.io.manifest import config_hash
    assert manifest["config_hash"] == config_hash(manifest["config"])
    assert manifest["config"]["models"] == [QWEN2_5_7B.name]
    assert manifest["config"]["model_revisions"] == {
        QWEN2_5_7B.name: QWEN2_5_7B.revision,
    }
    assert manifest["config"]["baseline_dtype"] == "bfloat16"
    assert manifest["config"]["n_cdd_samples"] == 2
    assert manifest["config"]["cdd_score_definition"] == "actual-max-truncated-length-v2"
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

    # A pre-fix score directory must not silently accept the new definition.
    del manifest["config"]["cdd_score_definition"]
    manifest["config_hash"] = config_hash(manifest["config"])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="different run configuration"):
        run(config)


def test_run_records_decoding_settings_truncation_and_prompt_provenance(tmp_path, monkeypatch):
    """Paper §4.4's record list, end to end through run(): the resolved
    decoding settings in the manifest, the per-generation truncation flag, and
    the probability detector's target text / token boundaries / chat
    template."""
    import qcd.real_run as real_run_module
    from qcd.data.schema import PromptScoringDetail

    item = Item(
        item_id="q1", dataset=Dataset.LCB_PRE, prompt="fixed prompt",
        metadata={"contest_date": "2023-06-01"},
    )
    detail = PromptScoringDetail(
        logprobs=[-1.0, -2.0],
        target_token_indices=[4, 5],
        target_char_span=(22, 34),
        rendered_char_length=48,
        chat_template_applied=True,
        chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
        target_text="fixed prompt",
    )

    class FakeModel:
        tokenizer = object()
        max_new_tokens = 512

        def generate(self, item_id, prompt, *, temperature, sample_id):
            del item_id, prompt
            is_greedy = temperature == 0.0
            return SimpleNamespace(
                text="print(1)", token_ids=[1, 2], token_logprobs=[-0.2, -0.3],
                is_greedy=is_greedy,
                # Only the greedy reference output ran into the 512-token cap.
                truncated_at_cap=is_greedy,
            )

        def score_prompt_detail(self, item_id, prompt):
            del item_id, prompt
            return detail

    monkeypatch.setattr(real_run_module, "load_all_items", lambda config: [item])
    monkeypatch.setattr(
        real_run_module, "load_model", lambda spec, quant, mock=False: FakeModel()
    )
    monkeypatch.setattr(real_run_module, "_assemble_candidate_code", lambda item, text: text)
    monkeypatch.setattr(real_run_module, "partial_pass_rate", lambda item, code: 1.0)

    config = _small_config(
        tmp_path, n_cdd_samples=2, include_humaneval=False, include_mbppplus=False
    )
    run(config)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    decoding = manifest["config"]["decoding_settings"]
    assert decoding["greedy"]["follows_checkpoint_generation_config"] is False
    assert decoding["samples"]["resolved"]["top_p"] == 1.0
    assert decoding["samples"]["resolved"]["top_k"] == 0
    assert decoding["samples"]["resolved"]["repetition_penalty"] == 1.0
    assert decoding["samples"]["resolved"]["temperature"] == pytest.approx(0.8)
    assert decoding["greedy"]["resolved"]["do_sample"] is False
    assert decoding["greedy"]["resolved"]["max_new_tokens"] == 512
    assert manifest["config"]["generation_max_new_tokens"] == 512

    generations = pd.concat(
        [pd.read_parquet(path) for path in sorted((tmp_path / "raw").glob("generations*.parquet"))],
        ignore_index=True,
    )
    greedy = generations[generations["is_greedy"]].iloc[0]
    samples = generations[~generations["is_greedy"]]

    assert bool(greedy["truncated_at_cap"]) is True
    assert not samples["truncated_at_cap"].any()
    assert int(greedy["max_new_tokens"]) == 512
    assert greedy["decoding_settings_id"] == decoding["greedy_id"]
    assert set(samples["decoding_settings_id"]) == {decoding["samples_id"]}
    assert decoding["greedy_id"] != decoding["samples_id"]

    assert list(greedy["prompt_target_token_indices"]) == [4, 5]
    assert list(greedy["prompt_target_char_span"]) == [22, 34]
    assert greedy["prompt_target_text_sha256"] == detail.target_text_sha256
    assert bool(greedy["prompt_chat_template_applied"]) is True
    assert greedy["chat_template_id"] == detail.chat_template_id

    # The template text itself is stored once per run, not per item.
    templates = pd.read_parquet(tmp_path / "raw" / "chat_templates.parquet")
    assert len(templates) == 1
    assert templates.iloc[0]["chat_template_id"] == detail.chat_template_id
    assert "message.content" in templates.iloc[0]["chat_template"]
