"""Capture bead notebook stages and prepare its full CPU fit."""

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import time
from functools import partial
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from scipy.stats import qmc
from skimage.filters import gaussian
from threadpoolctl import threadpool_limits

from brighteyes import datasets
from brighteyes.mcs.reader import reader_legacy

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "examples/imaging/psf_distiller/distiller_test_beads.ipynb"


def text_sha256(path: Path) -> str:
    """Hash UTF-8 source text with checkout line endings normalized."""
    return hashlib.sha256(path.read_text("utf-8").encode("utf-8")).hexdigest()


def namespace(inputs: dict[str, np.ndarray]) -> dict:
    """Supply numerical inputs and module imports used by the bead notebook."""
    apr = import_module("brighteyes.imaging.analysis.APR_lib")
    return {
        "np": np,
        "gaussian": gaussian,
        "APR": apr.APR,
        "Reassignment": apr.Reassignment,
        "ShiftVectors": apr.ShiftVectors,
        "tool": import_module("brighteyes.imaging.analysis.Tools_lib"),
        "sim": import_module("brighteyes.imaging.simulation.PSF_sim"),
        "dis": import_module("brighteyes.imaging.psf_distiller.distiller"),
        "GridFinder": import_module(
            "brighteyes.imaging.s2ism.psf_estimator"
        ).GridFinder,
        "data": inputs["summed_image"],
        "clicked_points": inputs["clicked_points"],
        "n_beads": len(inputs["clicked_points"]),
    }


def numerical_source(cell: dict) -> str:
    """Keep computational statements while omitting notebook displays."""
    tree = ast.parse("".join(cell["source"]))
    statements = []
    for statement in tree.body:
        expression = ast.unparse(statement)
        if expression.startswith(
            (
                "_ = ShowDataset(",
                "fig = ShowDataset(",
                "fig, ax = plt.",
                "fig.",
                "ax[",
                "PlotShiftVectors(",
                "ShowFingerprint(",
                "plt.",
                "grid_fit.Print(",
            )
        ):
            continue
        statements.append(statement)
    tree.body = statements
    return ast.unparse(tree)


def save_stage(
    output: Path,
    name: str,
    cell: int,
    source: str,
    values: dict,
    allow_nan: bool = False,
) -> dict:
    """Save a numerical stage and its executable notebook statements."""
    arrays = {
        key: value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
        for key, value in values.items()
    }
    if any(
        array.dtype.hasobject
        or np.isinf(array).any()
        or (not allow_nan and np.isnan(array).any())
        for array in arrays.values()
    ):
        raise ValueError("stage arrays contain objects or unexpected nonfinite values")
    path = output / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    return {
        "name": name,
        "cell": cell,
        "executed_source": source,
        "variables": list(arrays),
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "nan_count": int(np.isnan(value).sum()),
            }
            for key, value in arrays.items()
        },
    }


def prepare(source: Path, output: Path) -> None:
    """Extract all selected raw beads and capture notebook preprocessing."""
    output.mkdir(parents=True, exist_ok=False)
    points_path = source.parent / "clicked_points.csv"
    points = np.loadtxt(points_path, delimiter="\t")
    if points.shape != (15, 2):
        raise ValueError("the notebook requires 15 saved bead coordinates")
    bounds = np.array(
        [
            [
                int(x + 100 - 18) - 100,
                int(x + 100 + 18) - 100,
                int(y + 100 - 18) - 100,
                int(y + 100 + 18) - 100,
            ]
            for x, y in points
        ]
    )
    with h5py.File(source, "r") as handle:
        raw = handle["data"]
        if raw.shape[:2] != (1, 1) or raw.shape[-1] != 25:
            raise ValueError("unexpected bead measurement axes")
        metadata = reader_legacy.MCSMetadata(handle)
        summed = np.empty(raw.shape[2:4] + (25,), dtype=np.uint64)
        for start in range(0, raw.shape[2], 32):
            summed[start : start + 32] = raw[0, 0, start : start + 32].sum(axis=-2)
        crops = np.zeros((15, 36, 36, raw.shape[-2], 25), dtype=raw.dtype)
        for index, (xmin, xmax, ymin, ymax) in enumerate(bounds):
            left, right = max(xmin, 0), min(xmax, raw.shape[3])
            top, bottom = max(ymin, 0), min(ymax, raw.shape[2])
            crops[index, top - ymin : bottom - ymin, left - xmin : right - xmin] = raw[
                0, 0, top:bottom, left:right
            ]
        raw_shape = list(raw.shape)
    inputs = {
        "raw_bead_crops": crops,
        "summed_image": summed,
        "clicked_points": points,
        "crop_bounds_xy": bounds,
        "pixel_size_um": np.asarray(metadata.dx),
        "pixel_dwell_us": np.asarray(metadata.pxdwelltime),
        "raw_crop_axis_order": np.asarray("byxtc"),
    }
    input_path = output / "beads-measured-small.npz"
    np.savez_compressed(input_path, **inputs)
    notebook = json.loads((ROOT / NOTEBOOK).read_text("utf-8"))
    context = namespace(inputs)
    scientific_sources = {
        str(Path(module.__file__).resolve().relative_to(ROOT)).replace(
            "\\", "/"
        ): text_sha256(Path(module.__file__))
        for module in (
            context["tool"],
            context["sim"],
            context["dis"],
            import_module(context["APR"].__module__),
            import_module(context["GridFinder"].__module__),
        )
    }
    stages = []
    for cell, name, variables in (
        (12, "padding", ("clicked_points_pad",)),
        (13, "bead-extraction", ("single_beads",)),
        (17, "registration-shifts", ("sum5", "shift_vec", "result_ism_pc")),
        (19, "registered-beads", ("single_beads_registered",)),
        (21, "summed-psf", ("exp_psf",)),
        (23, "centered-psf", ("data",)),
        (27, "background-subtraction", ("bg", "exp_psf_filtered")),
        (28, "detector-shifts", ("shift_vectors",)),
        (30, "detector-calibration", ("spad",)),
    ):
        source_text = numerical_source(notebook["cells"][cell])
        start = time.monotonic()
        exec(compile(source_text, f"bead-notebook-cell-{cell}", "exec"), context)  # noqa: S102
        stage = save_stage(
            output, name, cell, source_text, {key: context[key] for key in variables}
        )
        stage["source_sha256"] = hashlib.sha256(
            "".join(notebook["cells"][cell]["source"]).encode()
        ).hexdigest()
        stage["seconds"] = time.monotonic() - start
        stages.append(stage)
        print(f"Captured {name}: {stage['seconds']:.2f}s", flush=True)
    np.testing.assert_array_equal(crops.sum(axis=-2), context["single_beads"])
    registered_path = source.parent / "02_single_beads_registered.npy"
    historical = np.load(registered_path, allow_pickle=False)
    historical_comparison = {
        "sha256": hashlib.sha256(registered_path.read_bytes()).hexdigest(),
        "exact": bool(np.array_equal(historical, context["single_beads_registered"])),
        "max_absolute_difference": float(
            np.max(np.abs(historical - context["single_beads_registered"]))
        ),
    }
    manifest = {
        "status": "preprocessing-captured-full-fit-pending",
        "notebook": NOTEBOOK,
        "notebook_sha256": text_sha256(ROOT / NOTEBOOK),
        "text_hash_policy": "UTF-8 with LF line endings",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "scientific_sources_sha256": scientific_sources,
        "measurement": {
            "measurement_id": "measurement-002",
            "bytes": source.stat().st_size,
            "shape": raw_shape,
            "full_checksum_verified": False,
        },
        "coordinates_sha256": hashlib.sha256(points_path.read_bytes()).hexdigest(),
        "input": {
            "dataset": "beads-measured-small",
            "filename": input_path.name,
            "bytes": input_path.stat().st_size,
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        },
        "historical_registered_beads": historical_comparison,
        "environment": environment(),
        "adjustments": [
            "Read raw HDF5 in strips to limit memory use.",
            "Retain raw crops for all 15 beads, bins and channels.",
            "Omit notebook displays.",
        ],
        "stages": stages,
        "full_fit_source": numerical_source(notebook["cells"][31]),
        "post_fit_stages": [
            {
                "cell": cell,
                "name": name,
                "executed_source": numerical_source(notebook["cells"][cell]),
                "variables": variables,
            }
            for cell, name, variables in (
                (
                    33,
                    "aberration-coefficients",
                    ["zernike_output_ex", "zernike_output_em"],
                ),
                (35, "distilled-psfs", ["fit"]),
                (36, "wavefronts", ["wavefront_fit_ex", "wavefront_fit_em"]),
                (38, "vortex-psfs", ["psf_vortex"]),
            )
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def environment() -> dict:
    """Describe numerical dependencies without local filesystem paths."""
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "device": "cpu",
        "threads": 1,
        "packages": {
            name: version(name)
            for name in ("numpy", "scipy", "torch", "scikit-image", "psf-generator")
        },
    }


def full_fit(output: Path) -> None:
    """Run the original full fit from packaged inputs and save its results."""
    output.mkdir(parents=True, exist_ok=False)
    reference = ROOT / "tests/data/imaging/psf-distillation/beads"
    manifest = json.loads((reference / "manifest.json").read_text("utf-8"))
    for filename, checksum in manifest["scientific_sources_sha256"].items():
        if text_sha256(ROOT / filename) != checksum:
            raise ValueError(f"scientific source changed since preparation: {filename}")
    with np.load(datasets.fetch("beads-measured-small"), allow_pickle=False) as data:
        context = namespace({key: data[key] for key in data.files})
    for stage in manifest["stages"]:
        exec(  # noqa: S102
            compile(stage["executed_source"], f"cell-{stage['cell']}", "exec"), context
        )
        with np.load(reference / stage["file"], allow_pickle=False) as expected:
            for key in stage["variables"]:
                actual = context[key]
                if isinstance(actual, torch.Tensor):
                    actual = actual.detach().cpu().numpy()
                np.testing.assert_array_equal(actual, expected[key], strict=True)
    distiller = context["dis"]
    with patch.object(qmc, "Sobol", partial(qmc.Sobol, rng=0)):
        starts = distiller.generate_starting_points(28, 100, 2, "sobol")
    np.savez_compressed(output / "starting-points.npz", starting_points=starts.numpy())
    started = time.monotonic()
    minimize = distiller.minimizer
    completed = 0

    def record_start(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Save each completed optimizer start before returning its result."""
        nonlocal completed
        coefficients, losses = minimize(*args, **kwargs)
        np.savez_compressed(
            output / f"start-{completed:03d}.npz",
            coefficients=coefficients.detach().cpu().numpy(),
            history=losses.detach().cpu().numpy(),
        )
        completed += 1
        return coefficients, losses

    with (
        patch.object(distiller, "generate_starting_points", return_value=starts),
        patch.object(distiller, "minimizer", side_effect=record_start),
    ):
        exec(  # noqa: S102
            compile(manifest["full_fit_source"], "bead-notebook-cell-31", "exec"),
            context,
        )
    stage = save_stage(
        output,
        "full-fit",
        31,
        manifest["full_fit_source"],
        {key: context[key] for key in ("result", "history")},
    )
    stage["seconds"] = time.monotonic() - started
    stage["environment"] = environment()
    stage["adjustment"] = "Freeze the scrambled Sobol starting points with rng=0."
    stage["input_manifest_sha256"] = text_sha256(reference / "manifest.json")
    stage["text_hash_policy"] = manifest["text_hash_policy"]
    stage["source_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    stage["starting_points_sha256"] = hashlib.sha256(
        (output / "starting-points.npz").read_bytes()
    ).hexdigest()
    stage["post_fit_stages"] = []
    for entry in manifest["post_fit_stages"]:
        exec(  # noqa: S102
            compile(entry["executed_source"], f"cell-{entry['cell']}", "exec"), context
        )
        if entry["name"] == "wavefronts":
            coordinates = np.linspace(-1, 1, context["exPar"].mask_sampl)
            outside_pupil = np.hypot(coordinates[:, None], coordinates[None, :]) >= 1
            for key in entry["variables"]:
                np.testing.assert_array_equal(np.isnan(context[key]), outside_pupil)
        stage["post_fit_stages"].append(
            save_stage(
                output,
                entry["name"],
                entry["cell"],
                entry["executed_source"],
                {key: context[key] for key in entry["variables"]},
                allow_nan=entry["name"] == "wavefronts",
            )
        )
    (output / "manifest.json").write_text(
        json.dumps(stage, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Capture preprocessing from a measurement or run the prepared full fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    np.random.seed(0)
    torch.manual_seed(0)
    with threadpool_limits(limits=1):
        if args.source:
            prepare(args.source, args.output)
        else:
            full_fit(args.output)


if __name__ == "__main__":
    main()
