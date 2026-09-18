"""Paper §4.6's engineering-validation boundary, enforced in code.

§4.6: "Synthetic and smoke-test outputs are stored in a validation-only
namespace and are never used as manuscript evidence." These tests cover the
three places that has to hold: every manifest names its phase, the analysis
gate refuses anything that is not the main study, and the drivers write the
phase their namespace claims.

Also covers the two fixed constants of E-F11/E-F12: the common 2025-01-01
boundary as the CLI default, and the Llama role string matching §4.1/§4.2's
"diagnostic, not verification" wording.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from qcd.constants import LCB_SHARED_CONTROL_BOUNDARY
from qcd.io.manifest import (
    StudyPhase, build_manifest, coerce_study_phase, read_study_phase,
    require_main_study, write_manifest,
)
from qcd.models.registry import LLAMA3_1_8B, OLMO3_7B

_SCRIPTS = Path(__file__).parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_run(tmp_path: Path, study_phase) -> Path:
    write_manifest(build_manifest({"x": 1}, study_phase=study_phase), tmp_path / "manifest.json")
    return tmp_path


# --- study_phase is mandatory ----------------------------------------------


def test_build_manifest_requires_study_phase():
    with pytest.raises(TypeError):
        build_manifest({"x": 1})


def test_build_manifest_rejects_an_unknown_phase():
    with pytest.raises(ValueError, match="study_phase must be one of"):
        build_manifest({"x": 1}, study_phase="pilot")


def test_coerce_accepts_enum_and_its_value():
    assert coerce_study_phase(StudyPhase.MAIN_STUDY) == "main_study"
    assert coerce_study_phase("engineering_validation") == "engineering_validation"


# --- the analysis gate ------------------------------------------------------


def test_require_main_study_accepts_a_main_study_run(tmp_path):
    manifest = require_main_study(_write_run(tmp_path, StudyPhase.MAIN_STUDY))
    assert manifest["study_phase"] == "main_study"


def test_require_main_study_refuses_validation_output(tmp_path):
    run_dir = _write_run(tmp_path, StudyPhase.ENGINEERING_VALIDATION)
    with pytest.raises(ValueError, match="engineering_validation"):
        require_main_study(run_dir)


def test_require_main_study_accepts_the_manifest_file_itself(tmp_path):
    _write_run(tmp_path, StudyPhase.MAIN_STUDY)
    assert require_main_study(tmp_path / "manifest.json")["study_phase"] == "main_study"


def test_require_main_study_refuses_a_tree_without_a_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="no run manifest"):
        require_main_study(tmp_path)


def test_require_main_study_refuses_a_legacy_manifest_without_the_field(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no `study_phase`"):
        require_main_study(tmp_path)


def test_read_study_phase_reports_the_recorded_phase(tmp_path):
    assert read_study_phase(_write_run(tmp_path, StudyPhase.ENGINEERING_VALIDATION)) == (
        "engineering_validation"
    )


# --- the drivers write the phase their namespace claims ---------------------


def test_run_main_defaults_to_the_main_study_namespace():
    run_main = _load_script("run_main")
    assert run_main.MAIN_OUTPUT_DIR.is_absolute()
    # Repo-root-anchored, not CWD-relative (E-F3): `pipeline/data/...` is not
    # covered by .gitignore's root-anchored `/data/` and is not synced.
    repo_root = Path(__file__).resolve().parents[2]
    assert run_main.MAIN_OUTPUT_DIR == repo_root / "data" / "raw" / "main"


def test_run_main_defaults_the_common_boundary_to_the_constant():
    # E-F11: the boundary is a fixed design value, so the driver no longer
    # requires it to be typed at every invocation; the override stays for
    # §4.2's boundary-sensitivity re-runs.
    args = _load_script("run_main").build_parser().parse_args([])
    assert args.lcb_cutoff == LCB_SHARED_CONTROL_BOUNDARY
    assert args.output_dir == Path(__file__).resolve().parents[2] / "data" / "raw" / "main"


def test_smoke_test_writes_the_validation_namespace_and_phase():
    smoke = _load_script("run_smoke_test")
    repo_root = Path(__file__).resolve().parents[2]
    assert smoke._DATA_DIR == repo_root / "data" / "raw" / "validation" / "smoke_test"
    assert "engineering_validation" in smoke.__doc__


def test_lcb_smoke_test_writes_the_validation_namespace():
    lcb = _load_script("run_lcb_smoke_test")
    repo_root = Path(__file__).resolve().parents[2]
    assert lcb._DEFAULT_OUTPUT == repo_root / "data" / "raw" / "validation" / "lcb_smoke_test"


# --- E-F11 / E-F12 fixed strings -------------------------------------------


def test_common_boundary_is_fixed_at_the_paper_value():
    # Paper §4.2/§5 step 4: both Olmo Instruct cards state `Date cutoff:
    # Dec. 2024`, so 2025-01-01 is the first post-boundary date and the latest
    # of the five arms' bounds.
    assert LCB_SHARED_CONTROL_BOUNDARY == "2025-01-01"
    assert OLMO3_7B.primary_first_post_boundary == LCB_SHARED_CONTROL_BOUNDARY


def test_llama_role_does_not_claim_a_verified_cutoff():
    # Paper §4.1 names this role "external knowledge-boundary diagnostic"; the
    # LLMLagBench changepoint is an independent behavioral diagnostic of the
    # declaration, not a verification of it.
    role = LLAMA3_1_8B.role
    assert "verified" not in role.lower()
    assert "knowledge-boundary diagnostic" in role
