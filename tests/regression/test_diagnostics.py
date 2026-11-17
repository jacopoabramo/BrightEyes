"""Check numerical failure reports without changing comparison semantics."""

import json

import numpy as np
import pytest
from _diagnostics import assert_reference, record_environment


def test_report_preserves_unsigned_difference_and_arrays(tmp_path) -> None:
    """Report signed differences without wrapping unsigned arithmetic."""
    actual = np.array([0, 4], dtype=np.uint8)
    expected = np.array([10, 4], dtype=np.uint8)
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        assert_reference(actual, expected, "unsigned", directory=tmp_path)
    report = json.loads((tmp_path / "unsigned.json").read_text("utf-8"))
    assert report["max_absolute_error"] == 10
    assert report["max_error_fraction_of_reference_peak"] == 1
    assert report["largest_error_index"] == [0]
    assert report["different_values"] == 1
    with np.load(tmp_path / "unsigned.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["actual"], actual, strict=True)
        np.testing.assert_array_equal(saved["expected"], expected, strict=True)


def test_report_keeps_dtype_mismatch_a_failure(tmp_path) -> None:
    """Reject changed integer width even when numerical values match."""
    with pytest.raises(AssertionError, match="dtype"):
        assert_reference(
            np.array([4], dtype=np.int64),
            np.array([4], dtype=np.int32),
            "dtype",
            directory=tmp_path,
        )
    report = json.loads((tmp_path / "dtype.json").read_text("utf-8"))
    assert not report["exact"]
    assert report["different_values"] == 0
    assert report["actual_dtype"] != report["expected_dtype"]


def test_report_accepts_matching_nan_mask_and_zero_scale(tmp_path) -> None:
    """Preserve matching masked values without inventing a relative error."""
    values = np.array([np.nan, 0.0])
    assert_reference(values, values.copy(), "masked", directory=tmp_path)
    report = json.loads((tmp_path / "masked.json").read_text("utf-8"))
    assert report["exact"]
    assert report["max_error_fraction_of_reference_peak"] is None
    assert report["actual_nonfinite"] == 1


def test_report_preserves_shape_mismatch(tmp_path) -> None:
    """Record changed dimensions without masking the assertion failure."""
    with pytest.raises(AssertionError, match="shapes"):
        assert_reference(np.ones(2), np.ones((1, 2)), "shape", directory=tmp_path)
    report = json.loads((tmp_path / "shape.json").read_text("utf-8"))
    assert report["actual_shape"] != report["expected_shape"]
    with np.load(tmp_path / "shape.npz", allow_pickle=False) as saved:
        assert saved["actual"].shape == (2,)
        assert saved["expected"].shape == (1, 2)


def test_environment_omits_local_paths_and_unrelated_variables(tmp_path, monkeypatch):
    """Exclude private paths and unrelated environment values from artifacts."""
    monkeypatch.setenv("SECRET_TOKEN", "must-not-be-exported")
    monkeypatch.setattr(
        "_diagnostics.threadpool_info",
        lambda: [{"filepath": "private-home-directory", "num_threads": 1}],
    )
    record_environment(tmp_path)
    content = (tmp_path / "environment.json").read_text("utf-8")
    assert "private-home-directory" not in content
    assert "must-not-be-exported" not in content


@pytest.mark.parametrize("name", ("focus_ism_in", "focus_ism_out"))
def test_focus_maps_accept_only_subcount_variation(name, tmp_path) -> None:
    """Accept the approved count margin while retaining exact upstream checks."""
    actual = np.array([0.10722331031598298, 595.0])
    expected = np.array([0.053058485309533165, 595.0])
    label = f"focus-ism/{name}"
    assert_reference(actual, expected, label, directory=tmp_path)
    report = json.loads((tmp_path / f"focus-ism-{name}.json").read_text("utf-8"))
    assert report["passed"]
    assert not report["exact"]
    with pytest.raises(AssertionError):
        assert_reference(actual, expected, "apr/apr")
    with pytest.raises(AssertionError):
        assert_reference(np.array([0.101]), np.array([0.0]), label)


@pytest.mark.parametrize("change", ("shape", "dtype"))
def test_tolerant_comparisons_preserve_array_structure(change) -> None:
    """Reject changed dimensions or precision even within a value tolerance."""
    expected = np.array([1.0], dtype=np.float64)
    match change:
        case "shape":
            actual = expected.reshape(1, 1)
        case "dtype":
            actual = expected.astype(np.float32)
    with pytest.raises(AssertionError):
        assert_reference(actual, expected, "focus-ism/focus_ism_in")


@pytest.mark.parametrize(
    "label, accepted, rejected",
    (
        ("bead-fit/loss-prefix", 1e-7, 1e-5),
        ("psf-distiller-simulation/result", 1e-7, 1e-5),
        ("ism-frc/result_0_2", 1e-8, 1e-7),
        ("ism-frc/result_0_4", 1e-8, 1e-7),
        ("ism-frc/result_1_2", 1e-8, 1e-7),
        ("ism-frc/result_1_4", 1e-8, 1e-7),
    ),
)
def test_float_limits_reject_larger_computational_changes(label, accepted, rejected):
    """Accept backend-scale rounding while rejecting larger numerical changes."""
    expected = np.array([1.0])
    assert_reference(expected + accepted, expected, label)
    with pytest.raises(AssertionError):
        assert_reference(expected + rejected, expected, label)
