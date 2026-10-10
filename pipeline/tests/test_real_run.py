"""Real-network, GPU-free tests for qcd.real_run — the shared core behind
scripts/run_main.py. Confirms the parts of the
real (non-mock) pipeline that don't need a GPU: item loading/capping,
candidate-code assembly, and that `run()` fails at exactly the expected
place (model loading) rather than somewhere earlier due to a wiring bug —
leaving items.parquet and manifest.json already written when it does.
"""

import concurrent.futures
import dataclasses
import datetime as dt
import json
import os
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from qcd.config import ModelSpec, Quant
from qcd.data.schema import Dataset, Item
from qcd.models.registry import QWEN2_5_7B
from qcd.real_run import RealRunConfig, _assemble_candidate_code, _generation_prompt, load_all_items, run
from tests import sandbox_fakes


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

        def generate_samples(self, item_id, prompt, *, temperature, sample_ids):
            return [
                self.generate(item_id, prompt, temperature=temperature, sample_id=sample_id)
                for sample_id in sample_ids
            ]

        def score_prompt_logprobs(self, item_id, prompt):
            self.prompt_calls.append((item_id, prompt))
            return [-1.0, -2.0, -3.0]

    fake_model = FakeModel()
    monkeypatch.setattr(real_run_module, "load_all_items", lambda config: [item])
    monkeypatch.setattr(
        real_run_module, "load_model", lambda spec, quant, mock=False: fake_model
    )
    monkeypatch.setattr(real_run_module, "_assemble_candidate_code", lambda item, text: text)
    monkeypatch.setattr(real_run_module, "_timed_partial_pass_rate", sandbox_fakes.passes)

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

        def generate_samples(self, item_id, prompt, *, temperature, sample_ids):
            return [
                self.generate(item_id, prompt, temperature=temperature, sample_id=sample_id)
                for sample_id in sample_ids
            ]

        def score_prompt_detail(self, item_id, prompt):
            del item_id, prompt
            return detail

    monkeypatch.setattr(real_run_module, "load_all_items", lambda config: [item])
    monkeypatch.setattr(
        real_run_module, "load_model", lambda spec, quant, mock=False: FakeModel()
    )
    monkeypatch.setattr(real_run_module, "_assemble_candidate_code", lambda item, text: text)
    monkeypatch.setattr(real_run_module, "_timed_partial_pass_rate", sandbox_fakes.passes)

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
    templates = pd.read_parquet(tmp_path / "raw" / "chat_templates.Qwen2.5-7B-Instruct-bf16.parquet")
    assert len(templates) == 1
    assert templates.iloc[0]["chat_template_id"] == detail.chat_template_id
    assert "message.content" in templates.iloc[0]["chat_template"]


class _CountingModel:
    tokenizer = object()

    def __init__(self, fail_after=None):
        self.calls = []
        self.fail_after = fail_after

    def _record(self, call):
        if self.fail_after is not None and len(self.calls) == self.fail_after:
            raise KeyboardInterrupt("simulated interruption")
        self.calls.append(call)

    @staticmethod
    def _sample(item_id, sample_id, temperature):
        token = int(item_id[1:]) * 100 + sample_id
        return SimpleNamespace(
            text=f"print({token})", token_ids=[token],
            token_logprobs=[-0.5], is_greedy=temperature == 0.0,
        )

    def generate(self, item_id, prompt, *, temperature, sample_id):
        self._record((item_id, "greedy", (sample_id,)))
        return self._sample(item_id, sample_id, temperature)

    def generate_samples(self, item_id, prompt, *, temperature, sample_ids):
        self._record((item_id, "samples", tuple(sample_ids)))
        return [self._sample(item_id, s, temperature) for s in sample_ids]

    def score_prompt_logprobs(self, item_id, prompt):
        return [-1.0]


@pytest.fixture
def run_with(monkeypatch, tmp_path):
    """Runs `run()` on fixed LCB items q1..qN (two by default) against the
    given model, with n=3 samples drawn at the given batch size, scored by a
    `sandbox_fakes` function in the driver's spawned sandbox workers."""
    import qcd.real_run as real_run_module

    monkeypatch.setattr(real_run_module, "_assemble_candidate_code", lambda item, text: text)
    monkeypatch.setenv("QCD_TEST_SANDBOX_LOG", str(tmp_path / "sandbox.log"))

    def _load(model):
        def load(spec, quant, mock=False):
            _run.loads.append((spec.name, quant))
            return model
        return load

    def _run(
        model, output_dir, *, batch_size=2, spec=_TEST_QWEN, n_items=2,
        sandbox=sandbox_fakes.passes, **overrides,
    ):
        items = [
            Item(item_id=f"q{i}", dataset=Dataset.LCB_PRE, prompt=f"prompt {i}",
                 metadata={"contest_date": "2023-06-01"})
            for i in range(1, n_items + 1)
        ]
        monkeypatch.setattr(real_run_module, "load_all_items", lambda config: items)
        monkeypatch.setattr(real_run_module, "_timed_partial_pass_rate", sandbox)
        monkeypatch.setattr(real_run_module, "load_model", _load(model))
        run(_small_config(
            output_dir, n_cdd_samples=3, include_humaneval=False, include_mbppplus=False,
            models=(dataclasses.replace(spec, sample_batch_size=batch_size),),
            **overrides,
        ))

    _run.loads = []
    return _run


def _read_raw(output_dir, kind):
    """One raw table across its parts, without the wall-clock timing columns,
    which no two runs share."""
    frame = pd.concat(
        [pd.read_parquet(p) for p in sorted((output_dir / "raw").glob(f"{kind}.*.parquet"))],
        ignore_index=True,
    )
    keys = (
        ["quant", "item_id", "sample_id"] if kind == "generations"
        else ["quant", "item_id", "detector"]
    )
    return frame.drop(columns=[c for c in frame.columns if c.endswith("_seconds")]) \
        .sort_values(keys).reset_index(drop=True)


def _sandbox_log(tmp_path):
    return (tmp_path / "sandbox.log").read_text().splitlines()


def _part_bytes(output_dir, pattern="*.parquet"):
    return {p.name: p.read_bytes() for p in sorted((output_dir / "raw").glob(pattern))}


def test_an_interrupted_run_resumes_from_cache_to_the_same_output(tmp_path, run_with):
    # Interrupted between q2's two sample chunks: q1 is complete, q2's greedy
    # output is cached, and q2's first chunk was generated but not cached.
    interrupted = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run_with(_CountingModel(fail_after=5), interrupted)

    resumed = _CountingModel()
    run_with(resumed, interrupted)
    # Only q2's sample set is generated again, whole, in the same chunks.
    assert resumed.calls == [("q2", "samples", (1, 2)), ("q2", "samples", (3,))]

    uninterrupted = tmp_path / "uninterrupted"
    run_with(_CountingModel(), uninterrupted)
    for kind in ("generations", "detector_scores"):
        pd.testing.assert_frame_equal(_read_raw(interrupted, kind), _read_raw(uninterrupted, kind))


def test_a_cache_made_at_another_batch_size_is_not_served(tmp_path, run_with):
    import shutil

    first = tmp_path / "batch2"
    run_with(_CountingModel(), first, batch_size=2)
    second = tmp_path / "batch1"
    shutil.copytree(first / "cache", second / "cache")

    model = _CountingModel()
    run_with(model, second, batch_size=1)

    assert [call for call in model.calls if call[1] == "samples"] == [
        (item, "samples", (s,)) for item in ("q1", "q2") for s in range(1, 4)
    ]
    manifest = json.loads((second / "manifest.json").read_text())
    assert manifest["config"]["sample_batch_sizes"] == {QWEN2_5_7B.name: 1}
    assert manifest["config"]["generation_seed_policy"] == (
        "sha256(item_id,sample_id,temperature)-per-row-generator-fixed-batch-v4"
    )


def test_a_model_without_a_measured_batch_size_refuses_before_loading(tmp_path, monkeypatch):
    import qcd.real_run as real_run_module
    from qcd.io.manifest import StudyPhase
    from qcd.models.registry import MAIN_ANALYSIS_MODELS

    loaded = []
    monkeypatch.setattr(
        real_run_module, "load_model", lambda spec, quant, mock=False: loaded.append(spec)
    )
    monkeypatch.setattr(
        real_run_module, "load_all_items", lambda config: loaded.append("items")
    )

    # What scripts/run_main.py builds before the H100 measurement: the registry
    # roster with both 32B models unset, main study.
    roster = tuple(
        dataclasses.replace(m, sample_batch_size=None) if "32B" in m.name else m
        for m in MAIN_ANALYSIS_MODELS
    )
    with pytest.raises(ValueError, match="Measure it on the H100") as refused:
        run(_small_config(tmp_path, models=roster, study_phase=StudyPhase.MAIN_STUDY))
    assert str(refused.value).startswith("['Qwen2.5-32B-Instruct', 'Olmo3.1-32B-Instruct'] have no")
    assert "scripts/measure_sample_batch.py" in str(refused.value)
    assert "models/registry.py" in str(refused.value)
    assert loaded == []
    assert not (tmp_path / "manifest.json").exists()


def test_a_validation_run_proceeds_with_an_explicit_batch_size(tmp_path, run_with):
    from qcd.io.manifest import StudyPhase
    from qcd.models.registry import QWEN2_5_32B

    model = _CountingModel()
    run_with(
        model, tmp_path, batch_size=2, study_phase=StudyPhase.ENGINEERING_VALIDATION,
        spec=dataclasses.replace(
            QWEN2_5_32B, sample_batch_size=None, primary_first_post_boundary="2023-11-01",
        ),
    )

    assert ("q1", "samples", (1, 2)) in model.calls
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["study_phase"] == "engineering_validation"
    assert manifest["config"]["sample_batch_sizes"] == {QWEN2_5_32B.name: 2}


# --- one (model, precision) cell per process --------------------------------

_BOTH = (Quant.BF16, Quant.BNB_NF4)
_QWEN = QWEN2_5_7B.name


def _manifest(output_dir):
    return json.loads((output_dir / "manifest.json").read_text())


def test_cell_processes_share_the_full_studys_manifest_and_output(tmp_path, run_with):
    full = tmp_path / "full"
    run_with(_CountingModel(), full, quant_levels=_BOTH)
    run_with.loads.clear()

    split = tmp_path / "split"
    run_with(_CountingModel(), split, quant_levels=_BOTH, cells=frozenset({(_QWEN, Quant.BF16)}))
    assert _manifest(split)["config_hash"] == _manifest(full)["config_hash"]
    assert run_with.loads == [(_QWEN, Quant.BF16)]
    assert set(_read_raw(split, "detector_scores")["quant"]) == {"bf16"}
    bf16_files = _part_bytes(split)

    # The second cell's process is accepted by the directory the first wrote,
    # and leaves the first cell's files exactly as they were.
    run_with(_CountingModel(), split, quant_levels=_BOTH, cells=frozenset({(_QWEN, Quant.BNB_NF4)}))
    assert run_with.loads == [(_QWEN, Quant.BF16), (_QWEN, Quant.BNB_NF4)]
    after = _part_bytes(split)
    for name, content in bf16_files.items():
        if "bf16" in name:
            assert after[name] == content, name
    for kind in ("generations", "detector_scores"):
        pd.testing.assert_frame_equal(_read_raw(split, kind), _read_raw(full, kind))


def test_an_unknown_cell_is_refused_before_loading(tmp_path, monkeypatch):
    import qcd.real_run as real_run_module

    loaded = []
    monkeypatch.setattr(
        real_run_module, "load_model", lambda spec, quant, mock=False: loaded.append(spec)
    )
    with pytest.raises(ValueError, match=r"Qwen2.5-7B-Instruct:bnb_nf4"):
        run(_small_config(tmp_path, cells=frozenset({(_QWEN, Quant.BNB_NF4)})))
    assert loaded == []
    assert not (tmp_path / "manifest.json").exists()


def test_a_completed_cell_is_skipped_without_loading_its_model(tmp_path, run_with):
    run_with(_CountingModel(), tmp_path)
    marker = tmp_path / "cells" / "Qwen2.5-7B-Instruct-bf16" / "complete.json"
    record = json.loads(marker.read_text())
    assert record["n_items"] == 2
    assert record["parts"] == ["Qwen2.5-7B-Instruct-bf16-00000"]
    assert record["config_hash"] == _manifest(tmp_path)["config_hash"]
    parts = _part_bytes(tmp_path, "[gd]*.parquet")
    assert run_with.loads == [(_QWEN, Quant.BF16)]

    recorded = marker.read_text()
    model = _CountingModel()
    run_with(model, tmp_path)
    assert run_with.loads == [(_QWEN, Quant.BF16)]
    assert model.calls == []
    assert _part_bytes(tmp_path, "[gd]*.parquet") == parts
    assert marker.read_text() == recorded

    # Interrupted after its last part but before the marker: every part is on
    # disk, so the marker is written without loading the model either.
    marker.unlink()
    run_with(model, tmp_path)
    assert run_with.loads == [(_QWEN, Quant.BF16)]
    assert json.loads(marker.read_text())["parts"] == record["parts"]


def test_an_interrupted_cell_resumes_at_its_first_unwritten_part(tmp_path, run_with, monkeypatch):
    import qcd.real_run as real_run_module

    monkeypatch.setattr(real_run_module, "_RAW_BATCH_ITEMS", 1)

    interrupted = tmp_path / "interrupted"
    # Three model calls per item (greedy, two sample chunks): the fourth is
    # q2's greedy output, after q1's part was flushed.
    with pytest.raises(KeyboardInterrupt):
        run_with(_CountingModel(fail_after=3), interrupted, sandbox=sandbox_fakes.records)
    first_part = _part_bytes(interrupted)
    assert sorted(first_part) == [
        "detector_scores.Qwen2.5-7B-Instruct-bf16-00000.parquet",
        "generations.Qwen2.5-7B-Instruct-bf16-00000.parquet",
        "items.parquet", "model_item_labels.parquet",
    ]
    assert not (interrupted / "cells" / "Qwen2.5-7B-Instruct-bf16" / "complete.json").exists()

    (tmp_path / "sandbox.log").unlink()
    resumed = _CountingModel()
    run_with(resumed, interrupted, sandbox=sandbox_fakes.records)
    # q1 is neither generated, sandboxed nor rewritten; only q2 runs.
    assert resumed.calls == [("q2", "greedy", (0,)), ("q2", "samples", (1, 2)), ("q2", "samples", (3,))]
    assert [line.split()[0] for line in _sandbox_log(tmp_path)] == ["q2"]
    after = _part_bytes(interrupted)
    for name, content in first_part.items():
        assert after[name] == content, name

    uninterrupted = tmp_path / "uninterrupted"
    run_with(_CountingModel(), uninterrupted, sandbox=sandbox_fakes.records)
    assert sorted(_part_bytes(interrupted)) == sorted(_part_bytes(uninterrupted))
    for kind in ("generations", "detector_scores"):
        resumed_frame = _read_raw(interrupted, kind)
        keys = ["item_id", "sample_id"] if kind == "generations" else ["item_id", "detector"]
        assert not resumed_frame.duplicated(subset=keys).any()
        pd.testing.assert_frame_equal(resumed_frame, _read_raw(uninterrupted, kind))


class _FixedOutputModel(_CountingModel):
    """q1: a 120-token greedy output and samples of 0, 5 and 150 tokens.
    q2: every output empty."""

    _LENGTHS = {"q1": {0: 120, 1: 0, 2: 5, 3: 150}, "q2": {0: 0, 1: 0, 2: 0, 3: 0}}

    @classmethod
    def _sample(cls, item_id, sample_id, temperature):
        n =cls._LENGTHS[item_id][sample_id]
        return SimpleNamespace(
            text="x" * n, token_ids=[7] * n, token_logprobs=[-0.5] * n,
            is_greedy=temperature == 0.0,
        )


def test_cdd_rows_record_threshold_length_and_empty_outputs(tmp_path, run_with):
    run_with(_FixedOutputModel(), tmp_path)

    scores = _read_raw(tmp_path, "detector_scores")
    cdd = scores[scores["detector"] == "cdd"].set_index("item_id")
    # l is the longest output after truncation to 100 tokens, not the cap
    # itself and not the untruncated 150.
    assert cdd.loc["q1", "cdd_threshold_length"] == 100
    assert cdd.loc["q1", "cdd_n_empty_samples"] == 1
    assert cdd.loc["q1", "cdd_greedy_empty"] == False  # noqa: E712
    # All outputs empty: l=0, a zero threshold, and peakedness 1.0.
    assert cdd.loc["q2", "cdd_threshold_length"] == 0
    assert cdd.loc["q2", "cdd_n_empty_samples"] == 3
    assert cdd.loc["q2", "cdd_greedy_empty"] == True  # noqa: E712
    assert cdd.loc["q2", "score"] == 1.0

    others = scores[scores["detector"] != "cdd"]
    assert others[["cdd_threshold_length", "cdd_n_empty_samples", "cdd_greedy_empty"]].isna().all().all()


def test_a_stored_sample_row_is_regenerated_from_its_sample_id(tmp_path, run_with):
    """A raw row's `sample_id` is the id its seed came from: generating that
    id again gives the stored tokens. `_CountingModel` puts the id it was
    asked for into the token, so a row stored under a shifted id fails."""
    model = _CountingModel()
    run_with(model, tmp_path)

    generations = _read_raw(tmp_path, "generations")
    samples = generations[~generations["is_greedy"]]
    assert sorted(set(samples["sample_id"])) == [1, 2, 3]
    for row in samples.itertuples():
        [regenerated] = model.generate_samples(
            row.item_id, "", temperature=row.decoding_temperature, sample_ids=[row.sample_id],
        )
        assert list(row.token_ids) == regenerated.token_ids, (row.item_id, row.sample_id)

    cdd = _read_raw(tmp_path, "detector_scores").query("detector == 'cdd'")
    for source in cdd["source_sample_ids"]:
        assert list(source) == [0, 1, 2, 3]


# --- one frozen commit and environment per main study ------------------------


def test_a_main_study_cell_from_another_commit_is_refused(tmp_path, run_with, monkeypatch):
    import qcd.io.manifest as manifest_module

    run_with(_CountingModel(), tmp_path, quant_levels=_BOTH, cells=frozenset({(_QWEN, Quant.BF16)}))
    frozen = _manifest(tmp_path)["git_commit"]

    monkeypatch.setattr(manifest_module, "get_git_commit_hash", lambda repo_dir=None: "f" * 40)
    model = _CountingModel()
    with pytest.raises(RuntimeError, match=f"frozen at commit {frozen}") as refused:
        run_with(model, tmp_path, quant_levels=_BOTH, cells=frozenset({(_QWEN, Quant.BNB_NF4)}))
    assert "move the directory aside" in str(refused.value)
    assert model.calls == []
    assert run_with.loads == [(_QWEN, Quant.BF16)]


def test_a_main_study_cell_with_other_package_versions_is_refused(tmp_path, run_with, monkeypatch):
    import qcd.io.manifest as manifest_module

    run_with(_CountingModel(), tmp_path, quant_levels=_BOTH, cells=frozenset({(_QWEN, Quant.BF16)}))

    installed = manifest_module.get_installed_package_versions
    monkeypatch.setattr(
        manifest_module, "get_installed_package_versions",
        lambda packages=manifest_module.DEFAULT_TRACKED_PACKAGES: {
            **installed(packages), "transformers": "0.0.0-other",
        },
    )
    with pytest.raises(RuntimeError, match="transformers"):
        run_with(_CountingModel(), tmp_path, quant_levels=_BOTH, cells=frozenset({(_QWEN, Quant.BNB_NF4)}))
    assert run_with.loads == [(_QWEN, Quant.BF16)]


def test_a_validation_run_is_not_frozen_to_one_commit(tmp_path, run_with, monkeypatch):
    import qcd.io.manifest as manifest_module
    from qcd.io.manifest import StudyPhase

    phase = StudyPhase.ENGINEERING_VALIDATION
    run_with(_CountingModel(), tmp_path, quant_levels=_BOTH, study_phase=phase,
             cells=frozenset({(_QWEN, Quant.BF16)}))
    monkeypatch.setattr(manifest_module, "get_git_commit_hash", lambda repo_dir=None: "f" * 40)
    run_with(_CountingModel(), tmp_path, quant_levels=_BOTH, study_phase=phase,
             cells=frozenset({(_QWEN, Quant.BNB_NF4)}))
    assert run_with.loads == [(_QWEN, Quant.BF16), (_QWEN, Quant.BNB_NF4)]


def test_each_cell_records_the_commit_and_environment_that_produced_it(tmp_path, run_with):
    run_with(_CountingModel(), tmp_path)

    manifest = _manifest(tmp_path)
    cell_dir = tmp_path / "cells" / "Qwen2.5-7B-Instruct-bf16"
    for record_name in ("started.json", "complete.json"):
        record = json.loads((cell_dir / record_name).read_text())
        assert record["git_commit"] == manifest["git_commit"], record_name
        assert record["git_tracked_diff_sha256"] == manifest["git_tracked_diff_sha256"], record_name
        assert record["package_versions"] == manifest["package_versions"], record_name
        assert record["hostname"] == manifest["hostname"], record_name
        assert "gpu_name" in record, record_name
        assert record["config_hash"] == manifest["config_hash"], record_name
    started = json.loads((cell_dir / "started.json").read_text())
    assert isinstance(started["cpu_count"], int) and started["cpu_count"] >= 1
    assert len(started["load_average"]) == 3


def test_a_resumed_cell_keeps_its_first_start_record(tmp_path, run_with):
    interrupted = _CountingModel(fail_after=1)
    with pytest.raises(KeyboardInterrupt):
        run_with(interrupted, tmp_path)
    cell_dir = tmp_path / "cells" / "Qwen2.5-7B-Instruct-bf16"
    first_start = (cell_dir / "started.json").read_text()

    run_with(_CountingModel(), tmp_path)

    assert (cell_dir / "started.json").read_text() == first_start
    resumes = sorted(cell_dir.glob("resumed-*.json"))
    assert len(resumes) == 1
    assert json.loads(resumes[0].read_text())["started_at_utc"] > json.loads(first_start)["started_at_utc"]


def test_a_cell_with_other_scoring_environment_variables_is_refused(tmp_path, run_with, monkeypatch):
    """evalplus's memory cap and per-task timeout come from the environment
    and change pass/fail, so they are part of the study's configuration."""
    monkeypatch.setenv("EVALPLUS_MAX_MEMORY_BYTES", "4294967296")
    monkeypatch.delenv("EVALPLUS_TIMEOUT_PER_TASK", raising=False)
    run_with(_CountingModel(), tmp_path, quant_levels=_BOTH, cells=frozenset({(_QWEN, Quant.BF16)}))
    recorded = _manifest(tmp_path)["config"]["scoring_environment"]
    assert recorded["EVALPLUS_MAX_MEMORY_BYTES"] == "4294967296"
    assert recorded["EVALPLUS_TIMEOUT_PER_TASK"] is None

    monkeypatch.setenv("EVALPLUS_MAX_MEMORY_BYTES", "-1")
    with pytest.raises(RuntimeError, match="EVALPLUS_MAX_MEMORY_BYTES"):
        run_with(_CountingModel(), tmp_path, quant_levels=_BOTH, cells=frozenset({(_QWEN, Quant.BNB_NF4)}))
    assert run_with.loads == [(_QWEN, Quant.BF16)]


# --- sandbox scoring in spawned workers --------------------------------------


def _parts_in_file_order(output_dir):
    """Each part file as written, rows unsorted, without timing columns."""
    parts = {}
    for path in sorted((output_dir / "raw").glob("[gd]*.parquet")):
        frame = pd.read_parquet(path)
        parts[path.name] = frame.drop(columns=[c for c in frame.columns if c.endswith("_seconds")])
    return parts


def test_pooled_scoring_writes_what_in_process_scoring_writes(tmp_path, run_with, monkeypatch):
    import qcd.real_run as real_run_module

    monkeypatch.setattr(real_run_module, "_RAW_BATCH_ITEMS", 2)
    pooled = tmp_path / "pooled"
    run_with(_CountingModel(), pooled, n_items=5, sandbox=sandbox_fakes.records)
    logged = [line.split() for line in _sandbox_log(tmp_path)]
    assert str(os.getpid()) not in {pid for _, pid, _ in logged}
    assert {inherited for _, _, inherited in logged} == {"False"}, "workers were forked"

    monkeypatch.setattr(
        real_run_module, "ProcessPoolExecutor",
        lambda workers, mp_context, initializer: concurrent.futures.ThreadPoolExecutor(1),
    )
    in_process = tmp_path / "in_process"
    run_with(_CountingModel(), in_process, n_items=5, sandbox=sandbox_fakes.records)

    pooled_parts, in_process_parts = _parts_in_file_order(pooled), _parts_in_file_order(in_process)
    assert list(pooled_parts) == list(in_process_parts)
    assert len(pooled_parts) == 6
    for name, frame in pooled_parts.items():
        pd.testing.assert_frame_equal(frame, in_process_parts[name])
    greedy = _read_raw(pooled, "generations").query("is_greedy")
    assert list(greedy["partial_pass_rate"]) == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_rows_keep_item_order_when_earlier_items_finish_scoring_last(tmp_path, run_with):
    run_with(_CountingModel(), tmp_path, n_items=4, sandbox=sandbox_fakes.slower_for_earlier_items)

    assert [line.split()[0] for line in _sandbox_log(tmp_path)] == ["q4", "q3", "q2", "q1"]
    generations = pd.read_parquet(next((tmp_path / "raw").glob("generations.*.parquet")))
    assert list(generations["item_id"]) == [f"q{i}" for i in range(1, 5) for _ in range(4)]
    greedy = generations[generations["is_greedy"]]
    assert list(greedy["partial_pass_rate"]) == [0.1, 0.2, 0.3, 0.4]
    assert list(greedy["passed"]) == [False] * 4
    # The worker's own scoring time, not the main process's wait for q1.
    assert list(greedy["sandbox_scoring_seconds"]) == [sandbox_fakes.SECONDS] * 4


class _ClockedModel(_CountingModel):
    def __init__(self):
        super().__init__()
        self.greedy_started = {}

    def generate(self, item_id, prompt, *, temperature, sample_id):
        self.greedy_started[item_id] = time.time()
        return super().generate(item_id, prompt, temperature=temperature, sample_id=sample_id)


def test_a_parts_scores_all_return_before_the_next_part_is_generated(tmp_path, run_with, monkeypatch):
    import qcd.real_run as real_run_module

    monkeypatch.setattr(real_run_module, "_RAW_BATCH_ITEMS", 2)
    model = _ClockedModel()
    run_with(model, tmp_path, n_items=4, sandbox=sandbox_fakes.slower_for_earlier_items)

    scored_at = {item: float(at) for item, at in (line.split() for line in _sandbox_log(tmp_path))}
    assert model.greedy_started["q3"] > max(scored_at["q1"], scored_at["q2"])


def test_a_worker_exception_stops_the_cell_without_a_partial_part(tmp_path, run_with):
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="sandbox failed on q2"):
        run_with(_CountingModel(), failed, sandbox=sandbox_fakes.fails_on_q2)
    assert not list((failed / "raw").glob("[gd]*.parquet"))
    assert not (failed / "cells" / "Qwen2.5-7B-Instruct-bf16" / "complete.json").exists()

    run_with(_CountingModel(), failed, sandbox=sandbox_fakes.records)
    clean = tmp_path / "clean"
    run_with(_CountingModel(), clean, sandbox=sandbox_fakes.records)
    assert sorted(_part_bytes(failed)) == sorted(_part_bytes(clean))
    for kind in ("generations", "detector_scores"):
        pd.testing.assert_frame_equal(_read_raw(failed, kind), _read_raw(clean, kind))


def test_the_sandbox_worker_count_is_part_of_the_run_configuration(tmp_path, run_with, monkeypatch):
    import qcd.real_run as real_run_module

    four = tmp_path / "four"
    run_with(_CountingModel(), four)
    assert _manifest(four)["config"]["sandbox_workers"] == 4

    monkeypatch.setattr(real_run_module, "_SANDBOX_WORKERS", 2)
    with pytest.raises(RuntimeError, match="different run configuration"):
        run_with(_CountingModel(), four)
    two = tmp_path / "two"
    run_with(_CountingModel(), two)
    assert _manifest(two)["config"]["sandbox_workers"] == 2
    assert _manifest(two)["config_hash"] != _manifest(four)["config_hash"]


def _start_method(_):
    import multiprocessing  # noqa: PLC0415

    return multiprocessing.get_start_method()


def test_sandbox_workers_fork_their_own_test_processes():
    # evalplus starts one process per test; under the inherited "spawn" each
    # re-imported the entry script and timed out reference solutions.
    from qcd.real_run import start_sandbox_pool  # noqa: PLC0415

    sandbox = start_sandbox_pool()
    try:
        assert sandbox.submit(_start_method, None).result() == "fork"
    finally:
        sandbox.shutdown()
