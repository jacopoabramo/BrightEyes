"""Compare frozen arrays with explicit limits for CPU-dependent outputs."""

import hashlib
import json
import platform
import re
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Final

import numpy as np
import torch
from threadpoolctl import threadpool_info

array_tolerances: Final[dict[str, tuple[float, float]]] = {
    "bead-fit/loss-prefix": (2 * float(np.finfo(np.float32).eps), 0.0),
    # Bounded fits on different BLAS paths split near-zero signals differently.
    "focus-ism/focus_ism_in": (0.0, 0.1),
    "focus-ism/focus_ism_out": (0.0, 0.1),
    "psf-distiller-simulation/result": (1e-6, 1e-9),  # Float32 CPU dispatch rounding.
    # The sigmoid fit changes slightly across BLAS CPU paths.
    "ism-frc/result_0_2": (0.0, 2e-8),
    "ism-frc/result_0_4": (0.0, 2e-8),
    "ism-frc/result_1_2": (0.0, 2e-8),
    "ism-frc/result_1_4": (0.0, 2e-8),
}


def record_environment(directory: Path) -> None:
    """Record numerical versions, CPU capabilities and active thread pools."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "environment.json"
    if destination.exists():
        return
    report = {
        "python": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "scipy",
                "torch",
                "scikit-image",
                "scikit-learn",
                "statsmodels",
                "psf-generator",
                "zernikepy",
                "h5py",
                "threadpoolctl",
            )
        },
        "torch": {
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "mkl_available": torch.backends.mkl.is_available(),
            "mkldnn_available": torch.backends.mkldnn.is_available(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
        },
        "thread_pools": [
            {
                key: pool[key]
                for key in (
                    "user_api",
                    "internal_api",
                    "num_threads",
                    "version",
                    "architecture",
                    "threading_layer",
                )
                if key in pool
            }
            for pool in threadpool_info()
        ],
    }
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def comparison_report(actual: np.ndarray, expected: np.ndarray) -> dict:
    """Measure array differences without choosing a numerical tolerance."""
    report = {
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        "actual_nonfinite": int((~np.isfinite(actual)).sum()),
        "expected_nonfinite": int((~np.isfinite(expected)).sum()),
    }
    if actual.shape != expected.shape:
        return {**report, "exact": False, "different_values": None}
    equal = (actual == expected) | (np.isnan(actual) & np.isnan(expected))
    report["different_values"] = int((~equal).sum())
    report["exact"] = bool(equal.all() and actual.dtype == expected.dtype)
    dtype = np.result_type(actual.dtype, expected.dtype, np.float64)
    current = actual.astype(dtype)
    reference = expected.astype(dtype)
    finite = np.isfinite(current) & np.isfinite(reference)
    finite_errors = np.abs(current[finite] - reference[finite])
    reference_values = np.abs(reference[np.isfinite(reference)])
    peak = float(reference_values.max()) if reference_values.size else 0.0
    maximum = float(finite_errors.max()) if finite_errors.size else None
    report.update(
        {
            "reference_peak": peak,
            "max_absolute_error": maximum,
            "rms_error": float(np.sqrt(np.mean(finite_errors**2)))
            if finite_errors.size
            else None,
            "max_error_fraction_of_reference_peak": maximum / peak
            if peak and maximum is not None
            else None,
        }
    )
    if finite_errors.size:
        errors = np.full(actual.shape, -np.inf)
        errors[finite] = finite_errors
        report["largest_error_index"] = list(
            map(int, np.unravel_index(np.argmax(errors), actual.shape))
        )
    return report


def assert_reference(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str,
    *,
    directory: Path | None = None,
) -> None:
    """Check values, shape and dtype with optional numerical diagnostics."""
    passed = False
    rtol, atol = array_tolerances.get(label, (0.0, 0.0))
    try:
        if label in array_tolerances:
            np.testing.assert_allclose(
                actual, expected, rtol=rtol, atol=atol, strict=True, err_msg=label
            )
        else:
            np.testing.assert_array_equal(actual, expected, strict=True, err_msg=label)
        passed = True
    finally:
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            report = comparison_report(actual, expected)
            report["passed"] = passed
            report["tolerance"] = {"rtol": rtol, "atol": atol}
            report["comparison"] = label
            name = re.sub(r"[^a-zA-Z0-9_-]+", "-", label).strip("-")
            (directory / f"{name}.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            if not report["exact"]:
                np.savez_compressed(
                    directory / f"{name}.npz", actual=actual, expected=expected
                )


def record_focus_inputs(
    image: np.ndarray, calibration: np.ndarray, directory: Path
) -> None:
    """Save fingerprint calibration and reassigned micro-images for focusISM."""
    focus = import_module("brighteyes.imaging.analysis.FocusISM_lib")
    fingerprint, model, parameters = focus.FitFingerprint(calibration)
    shifts, reassigned = focus.APR.APR(image, 10, image.shape[-1] // 2, filter_sigma=1)
    reassigned[reassigned < 0] = 0
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "focus-inputs.npz",
        fingerprint=fingerprint,
        model=model,
        parameters=parameters,
        shifts=shifts,
        reassigned=reassigned,
    )
