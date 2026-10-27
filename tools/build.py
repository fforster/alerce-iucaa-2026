"""Convert the jupytext sources in src/ to notebooks in the repository root and execute them.

    python tools/build.py                    # all notebooks
    python tools/build.py 03_supernovae      # some notebooks
    python tools/build.py --no-exec 03_...   # only convert

Options: --kernel NAME (default python3), --no-exec.
"""
import argparse
import pathlib
import time

import jupytext
import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("names", nargs="*")
parser.add_argument("--kernel", default="python3")
parser.add_argument("--no-exec", action="store_true")
args = parser.parse_args()

names = args.names or sorted(p.stem for p in (ROOT / "src").glob("*.py"))
for name in names:
    nb = jupytext.read(ROOT / "src" / f"{name}.py")
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.pop("jupytext", None)
    if not args.no_exec:
        t = time.time()
        client = NotebookClient(nb, timeout=900, kernel_name=args.kernel,
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute()
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        print(f"{name}: executed in {time.time() - t:.0f} s")
    nbformat.write(nb, ROOT / f"{name}.ipynb")
