"""Per-run manifest — git commit, config hash, installed-package versions,
seeds, timestamps (pipeline_build_plan.md's raw-data-schema section): the
record that lets a later re-analysis (or the boundary-sensitivity re-runs
of paper §4.2, or a reviewer) know exactly what produced a given
`data/raw/` tree.

Three further records live here because they are all "what actually
produced this output", and because every producer (real_run.py, the smoke
tests, scripts/quantize_model.py) and every consumer (the analysis code)
has to agree on one spelling of them:

  - `StudyPhase` / `require_main_study` — paper §4.6's engineering-validation
    boundary, enforced in code rather than by directory naming convention.
  - `resolve_library_defaults` — paper §4.3: "Settings described as left at a
    library default are not overridden by us; the resolved value is recorded
    in the run manifest, so reproduction reads the record rather than this
    sentence."
  - `calibration_overlap_report` / `require_calibration_overlap_report` —
    paper §4.3's required pre-execution check of AWQ calibration text against
    the evaluation prompts.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import hashlib
import inspect
import json
import platform
import os
import subprocess
import tempfile
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DEFAULT_TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy", "pandas", "pyarrow", "transformers", "torch", "bitsandbytes",
    "accelerate", "gptqmodel", "llmcompressor", "compressed-tensors",
    "evalplus", "datasets", "huggingface_hub", "statsmodels", "scipy",
)


class StudyPhase(enum.Enum):
    """Which of paper §4.6's two namespaces an output belongs to.

    §4.6: "Synthetic and smoke-test outputs are stored in a validation-only
    namespace and are never used as manuscript evidence." The phase is written
    into every run manifest so the separation is a checkable property of the
    data, not a directory-naming habit.
    """

    ENGINEERING_VALIDATION = "engineering_validation"
    MAIN_STUDY = "main_study"


STUDY_PHASE_VALUES: tuple[str, ...] = tuple(phase.value for phase in StudyPhase)

MANIFEST_FILENAME = "manifest.json"


def coerce_study_phase(study_phase: StudyPhase | str) -> str:
    """Accept either the enum or its string value; reject anything else.

    Deliberately strict: a typo'd phase string would silently create a third,
    unenforced namespace.
    """
    if isinstance(study_phase, StudyPhase):
        return study_phase.value
    if study_phase in STUDY_PHASE_VALUES:
        return str(study_phase)
    raise ValueError(
        f"study_phase must be one of {STUDY_PHASE_VALUES}, got {study_phase!r} — paper §4.6 "
        "recognizes exactly two namespaces (engineering validation, main study)."
    )


def get_git_commit_hash(repo_dir: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def get_installed_package_versions(packages: tuple[str, ...] = DEFAULT_TRACKED_PACKAGES) -> dict[str, str | None]:
    """Reads installed-distribution metadata (no import needed), so this is
    safe and cheap to call even for GPU packages not installed on the
    mock-only profile — they simply resolve to None."""
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def config_hash(config: dict) -> str:
    """Stable hash of a JSON-serializable config dict — `sort_keys=True` so
    key order never changes the hash, `default=str` so non-JSON-native
    values (enums, Paths) don't crash it."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclasses.dataclass
class RunManifest:
    study_phase: str
    git_commit: str | None
    config: dict
    config_hash: str
    package_versions: dict[str, str | None]
    seed: int | None
    timestamp_utc: str
    hostname: str
    platform: str
    python_version: str
    extra: dict = dataclasses.field(default_factory=dict)


def build_manifest(
    config: dict,
    *,
    study_phase: StudyPhase | str,
    packages: tuple[str, ...] = DEFAULT_TRACKED_PACKAGES,
    seed: int | None = None,
    repo_dir: str | Path | None = None,
    extra: dict | None = None,
) -> RunManifest:
    """`study_phase` is a required keyword: a manifest without it cannot be
    built, so no producer can write an output tree whose phase is unknown
    (paper §4.6)."""
    return RunManifest(
        study_phase=coerce_study_phase(study_phase),
        git_commit=get_git_commit_hash(repo_dir),
        config=dict(config),
        config_hash=config_hash(config),
        package_versions=get_installed_package_versions(packages),
        seed=seed,
        timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        hostname=platform.node(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        extra=extra or {},
    )


def write_manifest(manifest: RunManifest, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(manifest), f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path


def read_manifest(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


# --- Paper §4.6's engineering-validation boundary --------------------------


def manifest_path(path: str | Path) -> Path:
    """Accept either a run directory or the manifest file itself.

    A path that does not exist yet is read as a run directory, so the error a
    caller sees names the manifest it was looking for rather than the bare
    directory.
    """
    path = Path(path)
    if path.is_file() or path.suffix == ".json":
        return path
    return path / MANIFEST_FILENAME


def read_study_phase(path: str | Path) -> str:
    """The phase recorded for a run directory (or manifest file).

    Raises when the manifest is absent or carries no phase: an output tree
    whose phase cannot be established is never silently treated as study data.
    """
    resolved = manifest_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"no run manifest at {resolved} — paper §4.6 separates engineering-validation "
            "output from study data by the manifest's `study_phase`, so a tree without a "
            "manifest cannot be classified."
        )
    manifest = read_manifest(resolved)
    phase = manifest.get("study_phase")
    if phase is None:
        raise ValueError(
            f"{resolved} has no `study_phase` field — it was written before the phase became "
            "mandatory; re-generate it rather than assuming which namespace it belongs to."
        )
    return coerce_study_phase(phase)


def require_main_study(path: str | Path, *, consumer: str = "analysis") -> dict:
    """Gate for study-data consumers: return the manifest only when the run
    directory is main-study output.

    Paper §4.6: validation outputs "are never used as manuscript evidence" and
    "are not aggregated into detector effect sizes, proxy AUCs, base rates,
    cross-precision correlations, power estimates, or detector-ranking
    decisions". Analysis entry points call this before reading a raw tree.
    """
    resolved = manifest_path(path)
    phase = read_study_phase(resolved)
    if phase != StudyPhase.MAIN_STUDY.value:
        raise ValueError(
            f"{consumer} refused: {resolved} records study_phase={phase!r}, not "
            f"{StudyPhase.MAIN_STUDY.value!r}. Paper §4.6 — engineering-validation output is "
            "never used as manuscript evidence."
        )
    return read_manifest(resolved)


# --- Paper §4.3's "resolved value is recorded in the run manifest" ---------
#
# We do not re-fix any of these numbers in our own code (the user's decision:
# use the library default, but record what it resolved to). Each entry is
# either a value read out of the installed library / loaded model, or an
# explicit null plus the reason it could not be read on this machine. Nothing
# here falls back to a number typed from memory.


def _resolved(value, source: str) -> dict:
    return {"value": value, "source": source, "unavailable_reason": None}


def _unresolved(reason: str) -> dict:
    return {"value": None, "source": None, "unavailable_reason": reason}


def _signature_default(function, parameter: str):
    default = inspect.signature(function).parameters[parameter].default
    return None if default is inspect.Parameter.empty else default


def resolve_bnb_int8_outlier_threshold(model=None) -> dict:
    """`llm_int8_threshold` — paper §4.3's "the outlier threshold ... is left
    at the library default"."""
    quantization_config = getattr(getattr(model, "config", None), "quantization_config", None)
    threshold = getattr(quantization_config, "llm_int8_threshold", None)
    if threshold is not None:
        return _resolved(float(threshold), "loaded model's config.quantization_config")
    try:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415
    except ImportError as error:
        return _unresolved(f"transformers is not importable here: {error}")
    try:
        return _resolved(
            float(BitsAndBytesConfig(load_in_8bit=True).llm_int8_threshold),
            "transformers.BitsAndBytesConfig(load_in_8bit=True)",
        )
    except Exception as error:  # noqa: BLE001 - needs torch installed
        try:
            default = _signature_default(BitsAndBytesConfig.__init__, "llm_int8_threshold")
        except (KeyError, TypeError, ValueError):
            return _unresolved(f"could not construct or introspect BitsAndBytesConfig: {error}")
        if default is None:
            return _unresolved(f"could not construct or introspect BitsAndBytesConfig: {error}")
        return _resolved(
            float(default),
            "transformers.BitsAndBytesConfig.__init__ signature default "
            f"(instantiation unavailable here: {error})",
        )


def resolve_bnb_4bit_block_size(model=None) -> dict:
    """nf4 block size — paper §4.3's "block size left at the library default".

    Preferred source is a loaded 4-bit parameter's own quantization state; the
    fallback is the installed bitsandbytes' `Params4bit` signature default.
    """
    if model is not None:
        for parameter in getattr(model, "parameters", lambda: [])():
            if type(parameter).__name__ != "Params4bit":
                continue
            for attribute_owner, name in (
                (getattr(parameter, "quant_state", None), "blocksize"),
                (parameter, "blocksize"),
            ):
                block_size = getattr(attribute_owner, name, None)
                if block_size is not None:
                    return _resolved(int(block_size), "loaded model's bitsandbytes Params4bit state")
            break
    try:
        import bitsandbytes  # noqa: PLC0415
    except ImportError as error:
        return _unresolved(f"bitsandbytes is not installed here: {error}")
    try:
        default = _signature_default(bitsandbytes.nn.Params4bit.__init__, "blocksize")
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        return _unresolved(f"bitsandbytes.nn.Params4bit has no introspectable blocksize: {error}")
    if default is None:
        return _unresolved(
            "bitsandbytes.nn.Params4bit.blocksize defaults to None (resolved later, "
            "per device) — read it from a loaded 4-bit model instead"
        )
    return _resolved(int(default), "bitsandbytes.nn.Params4bit.__init__ signature default")


def resolve_bnb_modules_not_converted(model=None) -> dict:
    """The BNB skip list — paper §4.3's "the library's default skip list
    (language-model head)". Only a loaded model knows it: transformers derives
    it from the architecture (`quantizers/base.py::get_modules_to_not_convert`
    -> `get_keys_to_not_convert(model)`) when `llm_int8_skip_modules` is None.
    """
    if model is None:
        return _unresolved("requires a loaded bitsandbytes model — transformers resolves this per architecture")
    modules = getattr(getattr(model, "hf_quantizer", None), "modules_to_not_convert", None)
    if modules is None:
        return _unresolved(
            "loaded model exposes no hf_quantizer.modules_to_not_convert "
            "(not a bitsandbytes-quantized model, or transformers changed where it stores the list)"
        )
    return _resolved(sorted(str(name) for name in modules), "loaded model's hf_quantizer.modules_to_not_convert")


#: The llm-compressor / compressed-tensors preset `scripts/quantize_model.py`
#: names in its recipe; kept here so the group-size resolution and the recipe
#: cannot drift apart.
AWQ_PRESET_SCHEME = "W4A16_ASYM"


def resolve_awq_group_size(model=None) -> dict:
    """AWQ group size — paper §4.3's "group size left at the scheme's default"
    for `W4A16_ASYM`."""
    quantization_config = getattr(getattr(model, "config", None), "quantization_config", None)
    config_groups = getattr(quantization_config, "config_groups", None)
    if config_groups:
        sizes = sorted({
            getattr(getattr(group, "weights", None), "group_size", None)
            for group in config_groups.values()
        } - {None})
        if len(sizes) == 1:
            return _resolved(int(sizes[0]), "loaded AWQ checkpoint's config.quantization_config")
        if sizes:
            return _resolved([int(size) for size in sizes], "loaded AWQ checkpoint's config.quantization_config")
    try:
        from compressed_tensors.quantization.quant_scheme import PRESET_SCHEMES  # noqa: PLC0415
    except ImportError as error:
        return _unresolved(f"compressed-tensors is not installed here: {error}")
    scheme = PRESET_SCHEMES.get(AWQ_PRESET_SCHEME) or PRESET_SCHEMES.get(AWQ_PRESET_SCHEME.upper())
    group_size = getattr(getattr(scheme, "weights", None), "group_size", None)
    if group_size is None:
        return _unresolved(
            f"compressed-tensors preset {AWQ_PRESET_SCHEME!r} exposes no weights.group_size"
        )
    return _resolved(int(group_size), f"compressed_tensors PRESET_SCHEMES[{AWQ_PRESET_SCHEME!r}]")


def resolve_mbppplus_dataset_version() -> dict:
    """MBPP+ dataset version pin (paper §4.2's MBPP+ arm)."""
    try:
        from evalplus.data.mbpp import MBPP_PLUS_VERSION  # noqa: PLC0415
    except ImportError as error:
        return _unresolved(f"evalplus is not importable here: {error}")
    return _resolved(str(MBPP_PLUS_VERSION), "evalplus.data.mbpp.MBPP_PLUS_VERSION")


def resolve_library_defaults(*, model=None) -> dict[str, dict]:
    """Every §4.3 "left at the library default" setting, resolved.

    Pass a loaded model to fill in the entries only a model can answer (the
    BNB skip list, an actual nf4 block size, an AWQ checkpoint's group size);
    without one, those entries carry an explicit reason instead of a value.
    """
    return {
        "bnb_llm_int8_outlier_threshold": resolve_bnb_int8_outlier_threshold(model),
        "bnb_4bit_block_size": resolve_bnb_4bit_block_size(model),
        "bnb_modules_not_converted": resolve_bnb_modules_not_converted(model),
        "awq_group_size": resolve_awq_group_size(model),
        "mbppplus_dataset_version": resolve_mbppplus_dataset_version(),
    }


def unresolved_library_defaults(resolved: dict[str, dict]) -> dict[str, str]:
    """`{name: reason}` for every entry that could not be read — what a run
    has to report rather than guess."""
    return {
        name: entry["unavailable_reason"]
        for name, entry in resolved.items()
        if entry.get("unavailable_reason")
    }


# --- Paper §4.3's AWQ calibration/evaluation overlap check ------------------

CALIBRATION_OVERLAP_REPORT_FILENAME = "calibration_overlap_report.json"
#: 13-token n-grams, the same window `ground_truth/string_match.py` uses for
#: corpus matching, so the two lexical checks in this repo report comparable
#: evidence.
CALIBRATION_OVERLAP_NGRAM_SIZE = 13
CALIBRATION_OVERLAP_METHOD_ID = "lexical-token-ngram-v1"


def _overlap_method_record(ngram_size: int) -> dict:
    return {
        "method_id": CALIBRATION_OVERLAP_METHOD_ID,
        "normalization": (
            "NFKC, casefold, whitespace-collapsed "
            "(qcd.ground_truth.string_match.normalize_text)"
        ),
        "tokenization": (
            "identifier / number / single punctuation regex "
            "(qcd.ground_truth.string_match.tokenize)"
        ),
        "ngram_size": ngram_size,
        "direction": (
            "every evaluation text's n-gram set is looked up in the union of the "
            "selected calibration rows' n-grams"
        ),
        "short_text_rule": (
            "an evaluation text shorter than ngram_size tokens is instead checked for "
            "normalized-substring containment in a calibration row"
        ),
        "limitations": (
            "lexical only: it detects verbatim and near-verbatim reuse after "
            "normalization, and does not detect paraphrase, translation, or semantic "
            "equivalence. A zero count is not evidence of semantic non-overlap "
            "(paper §4.3)."
        ),
    }


def _ngrams(tokens: tuple[str, ...], n: int) -> set[tuple[str, ...]]:
    return {tokens[i : i + n] for i in range(max(0, len(tokens) - n + 1))}


def calibration_overlap_report(
    *,
    calibration_texts: Iterable[str],
    evaluation_texts: Iterable[tuple[str, str, str]],
    ngram_size: int = CALIBRATION_OVERLAP_NGRAM_SIZE,
    calibration: dict | None = None,
    max_examples: int = 100,
) -> dict:
    """Lexical overlap between AWQ calibration text and evaluation text.

    `evaluation_texts` yields `(source, key, text)` — e.g.
    `("livecodebench", "abc387_b", question_content)` or
    `("humaneval", "HumanEval/0", prompt)`.

    Paper §4.3: "Check candidates against all evaluation prompts and reference
    solutions before use; record search coverage and exclusions. This overlap
    check is a required pre-execution step, not a completed result or proof of
    semantic non-overlap."
    """
    from qcd.ground_truth.string_match import normalize_text, tokenize  # noqa: PLC0415

    if ngram_size < 1:
        raise ValueError("ngram_size must be positive")

    calibration_grams: set[tuple[str, ...]] = set()
    calibration_normalized: list[str] = []
    n_calibration_texts = 0
    for text in calibration_texts:
        n_calibration_texts += 1
        calibration_normalized.append(normalize_text(text))
        calibration_grams |= _ngrams(tokenize(text), ngram_size)

    per_source: dict[str, dict[str, int]] = {}
    overlaps: list[dict] = []
    n_evaluation_texts = 0
    n_overlapping = 0

    for source, key, text in evaluation_texts:
        n_evaluation_texts += 1
        counts = per_source.setdefault(source, {"n_texts": 0, "n_overlapping": 0})
        counts["n_texts"] += 1

        tokens = tokenize(text)
        if len(tokens) >= ngram_size:
            grams = _ngrams(tokens, ngram_size)
            matched = grams & calibration_grams
            if not matched:
                continue
            record = {
                "source": source,
                "key": key,
                "rule": "ngram",
                "n_matching_ngrams": len(matched),
                "ngram_coverage": len(matched) / len(grams),
                "example_ngram": " ".join(sorted(matched)[0]),
            }
        else:
            normalized = normalize_text(text)
            if not normalized or not any(
                normalized in candidate for candidate in calibration_normalized
            ):
                continue
            record = {
                "source": source,
                "key": key,
                "rule": "short-text-containment",
                "n_matching_ngrams": 0,
                "ngram_coverage": 1.0,
                "example_ngram": normalized[:200],
            }

        n_overlapping += 1
        counts["n_overlapping"] += 1
        if len(overlaps) < max_examples:
            overlaps.append(record)

    return {
        "method": _overlap_method_record(ngram_size),
        "calibration": dict(calibration or {}),
        "n_calibration_texts": n_calibration_texts,
        "n_calibration_ngrams": len(calibration_grams),
        "sources": [
            {"source": source, **counts} for source, counts in sorted(per_source.items())
        ],
        "n_evaluation_texts": n_evaluation_texts,
        "n_overlapping_texts": n_overlapping,
        "overlaps": overlaps,
        "overlaps_truncated": n_overlapping > len(overlaps),
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def write_calibration_overlap_report(report: dict, directory: str | Path) -> Path:
    path = Path(directory) / CALIBRATION_OVERLAP_REPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def require_calibration_overlap_report(checkpoint_dir: str | Path) -> dict:
    """Refuse an AWQ checkpoint whose §4.3 overlap check was never run.

    The check is a pre-execution requirement, so its absence — not a nonzero
    overlap count — is what blocks loading. A nonzero count is a finding to
    adjudicate and record, which the report itself carries.
    """
    path = Path(checkpoint_dir) / CALIBRATION_OVERLAP_REPORT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"AWQ checkpoint {checkpoint_dir} has no {CALIBRATION_OVERLAP_REPORT_FILENAME}. "
            "Paper §4.3 requires the calibration/evaluation-prompt overlap check before the "
            "checkpoint is used; re-run `python scripts/quantize_model.py <model-name>`, which "
            "writes the report next to the checkpoint."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    missing = [
        field for field in ("method", "n_evaluation_texts", "n_overlapping_texts")
        if field not in report
    ]
    if missing:
        raise ValueError(f"{path} is not a valid overlap report; missing fields: {missing}")
    if not report["n_evaluation_texts"]:
        raise ValueError(
            f"{path} records an overlap check that covered 0 evaluation texts — the check did "
            "not actually run against the evaluation prompts."
        )
    return report
