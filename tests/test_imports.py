import importlib
import pkgutil

import idocr


def test_all_modules_import():
    names = [m.name for m in pkgutil.walk_packages(idocr.__path__, prefix="idocr.")]
    assert names, "no modules discovered"
    for name in names:
        importlib.import_module(name)


def test_package_does_not_require_torch():
    import subprocess
    import sys

    # Run in a clean subprocess so prior tests importing torch don't pollute sys.modules
    res = subprocess.run(
        [sys.executable, "-c", "import idocr, sys; assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"idocr unexpectedly imported torch: {res.stderr}"
