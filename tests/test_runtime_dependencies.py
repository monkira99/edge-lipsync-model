from __future__ import annotations

import os
import subprocess
import sys


def test_colab_inline_matplotlib_backend_is_available() -> None:
    env = {
        **os.environ,
        "MPLBACKEND": "module://matplotlib_inline.backend_inline",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import matplotlib; print(matplotlib.get_backend())",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.strip() == "module://matplotlib_inline.backend_inline"
