"""
The plotting stack is the optional ``viz`` extra: a plain ``pip install h2mare``
must run the whole pipeline without it, and plotting without it must say which
extra to install rather than fail on a bare missing module.

Each check runs in a fresh interpreter with the viz packages blocked, so the
test process's own (installed) copies cannot mask an eager import.
"""

import subprocess
import sys
import textwrap

VIZ = ["cartopy", "IPython", "matplotlib", "plotly", "statsmodels"]

_BLOCK = f"""
import sys
for name in {VIZ!r}:
    sys.modules[name] = None  # "import name" now raises ModuleNotFoundError
"""


def _run(code: str, tmp_path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
    )


def test_the_pipeline_imports_without_the_viz_extra(tmp_path):
    proc = _run(
        """
        import h2mare
        import h2mare.cli
        import h2mare.pipeline_manager
        import h2mare.processing.compiler
        import h2mare.processing.extractor
        import h2mare.storage.parquet_indexer
        print("ok")
        """,
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("ok")


def test_plotting_without_the_viz_extra_names_it(tmp_path):
    proc = _run(
        """
        try:
            import h2mare.utils.plot
        except ImportError as e:
            print(e)
        """,
        tmp_path,
    )
    assert "pip install 'h2mare[viz]'" in proc.stdout, proc.stderr
    assert "cartopy is not installed" in proc.stdout


def test_the_indexer_plot_accessor_names_it_too(tmp_path):
    proc = _run(
        f"""
        from h2mare.storage.parquet_indexer import ParquetIndexer
        try:
            ParquetIndexer(r"{tmp_path}").plot
        except ImportError as e:
            print(e)
        """,
        tmp_path,
    )
    assert "pip install 'h2mare[viz]'" in proc.stdout, proc.stderr
