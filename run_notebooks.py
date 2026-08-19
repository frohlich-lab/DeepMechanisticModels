#!/usr/bin/env python3
"""Execute the paper notebooks and report which ones still run.

Local-only: the notebooks read the pipeline's outputs (eval/, res/, pretrain/)
and hit external APIs (Cellosaurus, cBioPortal, Cell Model Passports, Synapse).
None of that exists in CI, so this is a developer tool for checking that the
notebooks have not been broken by a refactor -- not a test suite.

Usage
-----
    ./venv/bin/python run_notebooks.py --list             # show what would run
    ./venv/bin/python run_notebooks.py                    # run all
    ./venv/bin/python run_notebooks.py figure_2 figure_3  # run matching ones
    ./venv/bin/python run_notebooks.py --save-notebooks   # keep executed copies

Exits non-zero if any notebook failed.

Figure output
-------------
The notebooks call ``savefig("fig2_panelA.pdf")`` with bare relative filenames,
which would litter the repository root. ``Figure.savefig`` is therefore patched
in the kernel to redirect every write into ``<out-dir>/figures/``, prefixed with
the notebook's stem, so ``figure_2.ipynb`` produces
``figure_2__fig2_panelA.pdf``. This also catches the ``fig.savefig(save_path)``
calls inside ``figures_paper/barplots.py`` and ``figures_paper/embeddings.py``,
since ``plt.savefig`` delegates to ``Figure.savefig``.

Notes
-----
Notebooks run with the *current* interpreter, not the kernel recorded in their
metadata -- a stale ``python3`` kernelspec pointing at another environment
otherwise produces import errors that look like real breakage. Invoke with the
interpreter that has the project's dependencies.

Working directory is the repository root, because ``figure_4.ipynb`` falls back
to ``Path("figures_paper/mmc2.xlsx")`` when ``__file__`` is undefined and
``cytof/data.py`` reads ``./data/cytof.csv``. Both the repository root and
``figures_paper/`` go on ``PYTHONPATH``, since the notebooks use both
``import figure_config`` and ``from figures_paper.de_corr_plots import ...``.

``DataFrame.to_csv``/``Series.to_csv`` are redirected the same way, into
``<out-dir>/data/`` -- ``figure_S7.ipynb`` writes a derived CSV that would
otherwise land in ``figures_paper/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SEARCH_ROOT = "figures_paper"
DEFAULT_OUT_DIR = "notebook_output"

# Cells idle this long are treated as hung. Generous: some notebooks refit.
DEFAULT_TIMEOUT = 1800

# Injected ahead of the notebook's own cells. Keeps every figure and derived
# data file the notebook writes inside the output directory, namespaced by
# notebook, so the repository never accumulates run artefacts.
REDIRECT_OUTPUTS = """\
import pathlib as _pl

import matplotlib
import pandas as _pd

matplotlib.use("Agg")
from matplotlib.figure import Figure as _Figure

_FIG_DIR = _pl.Path(r"{fig_dir}")
_DATA_DIR = _pl.Path(r"{data_dir}")
_PREFIX = "{prefix}"
_FIG_DIR.mkdir(parents=True, exist_ok=True)


def _redirect(target_dir, name):
    target_dir.mkdir(parents=True, exist_ok=True)
    return str(target_dir / f"{{_PREFIX}}__{{_pl.Path(name).name}}")


_orig_savefig = _Figure.savefig


def _redirected_savefig(self, fname, *args, **kwargs):
    # leave file objects / buffers alone, only rewrite paths
    if isinstance(fname, (str, _pl.PurePath)):
        fname = _redirect(_FIG_DIR, fname)
    return _orig_savefig(self, fname, *args, **kwargs)


_Figure.savefig = _redirected_savefig

# to_csv is inherited from NDFrame, so DataFrame and Series share one function;
# rebind on both. A None path means "return a string" -- leave that alone.
_orig_to_csv = _pd.DataFrame.to_csv


def _redirected_to_csv(self, path_or_buf=None, *args, **kwargs):
    if isinstance(path_or_buf, (str, _pl.PurePath)):
        path_or_buf = _redirect(_DATA_DIR, path_or_buf)
    return _orig_to_csv(self, path_or_buf, *args, **kwargs)


_pd.DataFrame.to_csv = _redirected_to_csv
_pd.Series.to_csv = _redirected_to_csv
"""


def discover(search_root: str, patterns: list[str]) -> list[Path]:
    """Return tracked notebooks under `search_root`, filtered by `patterns`."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", f"{search_root}/*.ipynb"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        found = [REPO_ROOT / p for p in out.split("\0") if p]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # not a git checkout, or git unavailable
        found = list((REPO_ROOT / search_root).rglob("*.ipynb"))

    found = [
        p for p in found if p.exists() and ".ipynb_checkpoints" not in p.parts
    ]
    if patterns:
        found = [
            p
            for p in found
            if any(pat in str(p.relative_to(REPO_ROOT)) for pat in patterns)
        ]
    return sorted(found, key=lambda p: str(p).lower())


def pinned_kernel_spec_manager():
    """KernelSpecManager that always resolves to the running interpreter.

    Avoids depending on a registered ``python3`` kernelspec, which need not
    point at the project's virtualenv.
    """
    from jupyter_client.kernelspec import KernelSpec, KernelSpecManager

    spec = KernelSpec(
        argv=[
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        display_name=f"runner ({sys.executable})",
        language="python",
    )

    class PinnedKernelSpecManager(KernelSpecManager):
        def get_kernel_spec(self, kernel_name):
            return spec

    return PinnedKernelSpecManager()


def _failure_report(nb, path: Path) -> tuple[str, str]:
    """Return (one-line summary, full report) for the first cell that errored.

    The one-liner alone is rarely enough to fix anything, so the full report
    carries the failing cell's source and traceback.
    """
    for idx, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") != "error":
                continue
            ename, evalue = out.get("ename", "?"), out.get("evalue", "")
            # cell 0 is the injected redirect, so report the notebook's own
            # numbering
            summary = f"{ename}: {evalue}  [cell {idx - 1}]"
            body = "\n".join(
                [
                    f"# {path}",
                    f"# failed in cell {idx - 1}: {ename}: {evalue}",
                    "",
                    "# ---- cell source ----",
                    "".join(cell.get("source", "")),
                    "",
                    "# ---- traceback ----",
                    *out.get("traceback", []),
                ]
            )
            return summary, body
    return "", ""


def run_one(
    path: Path,
    timeout: int,
    fig_dir: Path,
    data_dir: Path,
    save_to: Path | None,
    log_dir: Path,
) -> tuple[bool, str]:
    """Execute one notebook. Returns (ok, one-line message)."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    nb.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            REDIRECT_OUTPUTS.format(
                fig_dir=fig_dir, data_dir=data_dir, prefix=path.stem
            )
        ),
    )

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",  # remapped by the pinned spec manager
        kernel_spec_manager=pinned_kernel_spec_manager(),
        resources={"metadata": {"path": str(REPO_ROOT)}},
        # stop at the first failing cell; nbclient still records the error
        # output on that cell before raising, which _failure_report reads
        allow_errors=False,
    )
    ok, message = True, ""
    try:
        client.execute()
    except Exception as exc:  # cell error, kernel death, timeout, ...
        ok, message = False, f"{type(exc).__name__}: {exc}"

    if not ok:
        summary, body = _failure_report(nb, path)
        if not body:
            # Not every failure lands as an `error` output -- an unknown cell
            # magic, for instance, only writes a UsageError to stderr. Fall
            # back to the exception text and still surface a usable one-liner.
            body = f"# {path}\n# no error output recorded\n\n{message}"
            interesting = [
                ln.strip()
                for ln in message.split("\n")
                if ln.strip()
                and not ln.startswith("-")
                and "----- stderr -----" not in ln
                and not ln.startswith("An error occurred")
            ]
            summary = (
                interesting[-1] if interesting else message.split("\n")[0]
            )
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{path.stem}.log"
        log.write_text(body)
        message = f"{summary}\n          details: {log}"

    if save_to is not None:
        del nb.cells[0]  # drop the injected cell from the saved copy
        dest = save_to / path.relative_to(REPO_ROOT / DEFAULT_SEARCH_ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, dest)
    return ok, message


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "patterns",
        nargs="*",
        help="only run notebooks whose path contains one of these substrings",
    )
    parser.add_argument(
        "--search-root",
        default=DEFAULT_SEARCH_ROOT,
        help=f"directory to search (default: {DEFAULT_SEARCH_ROOT})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(DEFAULT_OUT_DIR),
        help=f"where figures land, under <out-dir>/figures "
        f"(default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"per-cell timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--save-notebooks",
        action="store_true",
        help="also write the executed notebooks to <out-dir>/executed/ "
        "(default: discard, leaving the tracked notebooks untouched)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="SUBSTRING",
        help="skip notebooks whose path contains this (repeatable)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first failure instead of running the rest",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="list the notebooks that would run, then exit",
    )
    args = parser.parse_args()

    notebooks = [
        p
        for p in discover(args.search_root, args.patterns)
        if not any(x in str(p) for x in args.exclude)
    ]
    if not notebooks:
        print(f"no notebooks found under {args.search_root}/")
        return 1

    if args.list_only:
        for p in notebooks:
            print(f"  {p.relative_to(REPO_ROOT)}")
        print(f"\n{len(notebooks)} notebook(s)")
        return 0

    missing = [
        d
        for d in ("nbformat", "nbclient", "ipykernel", "matplotlib")
        if importlib.util.find_spec(d) is None
    ]
    if missing:
        print(
            f"missing in {sys.executable}: {', '.join(missing)}\n"
            f"install with: {sys.executable} -m pip install "
            f"{' '.join(missing)}"
        )
        return 1

    out_dir = (
        args.out_dir
        if args.out_dir.is_absolute()
        else REPO_ROOT / args.out_dir
    )
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_to = out_dir / "executed" if args.save_notebooks else None
    log_dir = out_dir / "logs"

    # Both import styles the notebooks use must resolve; the kernel inherits
    # this environment.
    pypath = [str(REPO_ROOT), str(REPO_ROOT / args.search_root)]
    if existing := os.environ.get("PYTHONPATH", ""):
        pypath.append(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(pypath)
    os.environ.setdefault("MPLBACKEND", "Agg")  # no GUI windows

    print(f"interpreter : {sys.executable}")
    print(f"working dir : {REPO_ROOT}")
    print(f"figures     : {fig_dir}")
    print(f"notebooks   : {len(notebooks)}\n")

    results, t_all = [], time.monotonic()
    for i, path in enumerate(notebooks, 1):
        rel = path.relative_to(REPO_ROOT)
        print(f"[{i}/{len(notebooks)}] {rel} ... ", end="", flush=True)
        t0 = time.monotonic()
        ok, msg = run_one(
            path, args.timeout, fig_dir, data_dir, save_to, log_dir
        )
        print(f"{'ok' if ok else 'FAIL'} ({time.monotonic() - t0:.0f}s)")
        if not ok:
            print(f"          {msg}")
        results.append((rel, ok, msg))
        if not ok and args.fail_fast:
            print("\nstopping (--fail-fast)")
            break

    failed = [r for r in results if not r[1]]
    n_figs = len(list(fig_dir.glob("*")))
    print(f"\n{'-' * 62}")
    print(
        f"{len(results) - len(failed)}/{len(results)} passed in "
        f"{time.monotonic() - t_all:.0f}s, {n_figs} figure(s) in {fig_dir}"
    )
    if failed:
        print("\nfailed:")
        for rel, _, msg in failed:
            print(f"  {rel}\n      {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
