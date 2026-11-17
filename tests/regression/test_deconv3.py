"""Reproduce measured-data notebook stages from fixed raw inputs."""

import hashlib
import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from _diagnostics import record_focus_inputs
from _examples import numerical_arrays

from brighteyes import datasets


def test_deconv3_matches_saved_notebook_results(
    cpu_environment, reference_arrays, diagnostics_dir
) -> None:
    """Reproduce each measured-data notebook stage and its saved arrays."""
    raw_path = datasets.fetch("tomm20-deconv3-raw-small")
    psf_path = datasets.fetch("tomm20-deconv3-psf-small")
    reference = Path(__file__).parents[1] / "data" / "deconv3"
    manifest = json.loads((reference / "manifest.json").read_text("utf-8"))
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = archive["raw"]
    with np.load(psf_path, allow_pickle=False) as archive:
        psf = archive["psf_ism"]
    np.random.seed(0)
    torch.manual_seed(0)
    namespace = {
        "np": np,
        "torch": torch,
        "data": raw,
        "meta": SimpleNamespace(dx=manifest["raw_input"]["pixel_size_um"]),
        "pxsizex": manifest["raw_input"]["pixel_size_um"] * 1e3,
        "apr": import_module("brighteyes.imaging.analysis.APR_lib"),
        "fISM": import_module("brighteyes.imaging.analysis.FocusISM_lib"),
        "s2": import_module("brighteyes.imaging.s2ism.s2ism"),
        "pp": import_module("brighteyes.imaging.varreg.preprocessing"),
        "utils": import_module("brighteyes.imaging.varreg.utils"),
        "wp": import_module("brighteyes.imaging.varreg.whiteness_principle"),
        "psf_sim": import_module("brighteyes.imaging.simulation.PSF_sim"),
        "est": import_module("brighteyes.imaging.s2ism.psf_estimator"),
        "psf_ism": psf,
        "psf_in": psf[1],
        "psf_out": psf[0],
        "psf": psf[1].sum(-1),
        "psf_closed": psf[1, :, :, 12],
    }
    for stage in manifest["stages"]:
        if stage["name"] == "focus-ism" and diagnostics_dir is not None:
            size = namespace["Nx"]
            record_focus_inputs(
                namespace["dset"],
                namespace["dset"][
                    size // 4 : -size // 4 + 1, size // 4 : -size // 4 + 1, :
                ],
                diagnostics_dir,
            )
        expected_path = reference / stage["file"]
        assert hashlib.sha256(expected_path.read_bytes()).hexdigest() == stage["sha256"]
        exec(  # noqa: S102
            compile(stage["executed_source"], f"notebook-cell-{stage['cell']}", "exec"),
            namespace,
        )
        if stage["cell"] == 7:
            exec(  # noqa: S102
                compile(manifest["psf_capture"]["source"], "notebook-cell-13", "exec"),
                namespace,
            )
            assert (
                namespace["opt_out_of_focus_plane"]
                == manifest["psf_capture"]["optimal_depth_nm"]
            )
            reference_arrays(namespace["psf_ism"], psf, "psf-generation/psf_ism")
        actual = {
            key: array
            for name in stage["variables"]
            for key, array in numerical_arrays(namespace[name], name).items()
        }
        with np.load(expected_path, allow_pickle=False) as expected:
            assert set(actual) == set(expected.files)
            for key, array in actual.items():
                reference_arrays(array, expected[key], f"{stage['name']}/{key}")
