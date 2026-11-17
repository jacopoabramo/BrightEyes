"""Keep numerical regression checks on one CPU thread."""

import hashlib
import os
import random
import re
from collections.abc import Callable, Generator
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from _diagnostics import assert_reference, record_environment
from threadpoolctl import threadpool_limits


@pytest.fixture(autouse=True)
def cpu_environment() -> Generator[None, None, None]:
    """Restore random generators and CPU thread settings after each example."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.set_num_threads(threads)


@pytest.fixture
def diagnostics_dir(request, cpu_environment) -> Path | None:
    """Separate optional numerical artifacts by test without exporting paths."""
    output = os.environ.get("IMAGING_DIAGNOSTICS_DIR")
    if output is None:
        return None
    root = Path(output)
    record_environment(root)
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", request.node.name)[:48]
    suffix = hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:8]
    return root / f"{name}-{suffix}"


@pytest.fixture
def reference_arrays(diagnostics_dir) -> Callable:
    """Compare reference arrays and optionally export numerical diagnostics."""
    return partial(assert_reference, directory=diagnostics_dir)
