"""Check published notebooks for private filesystem locations."""

import json
import re
from pathlib import Path


def test_notebooks_omit_private_filesystem_locations() -> None:
    """Keep machine paths and internal server locations out of notebooks."""
    pattern = re.compile(
        r"(?<!\w)[A-Za-z]:[\\/]|\\\\[\w.-]+\\|/(?:home|Users)/"
        r"|Data MMS server|\.iit\.local",
        re.IGNORECASE,
    )
    root = Path(__file__).resolve().parents[2]
    for path in sorted((root / "examples").rglob("*.ipynb")):
        notebook = json.loads(path.read_text("utf-8"))
        pending = [notebook]
        while pending:
            value = pending.pop()
            match value:
                case dict():
                    pending.extend(
                        child
                        for key, child in value.items()
                        if key not in {"image/png", "image/jpeg"}
                    )
                case list():
                    pending.extend(value)
                case str():
                    assert not pattern.search(value), path.relative_to(root)
