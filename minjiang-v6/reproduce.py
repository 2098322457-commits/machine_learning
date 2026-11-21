"""Verify the final Nanping coursework model and produce its next-day forecast."""
import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V3 = ROOT / "minjiang-v3"
V4 = ROOT / "minjiang-v4"
DATA = ROOT / "minjiang-original" / "data"
DATA_COMMIT = "498ef586aee5baf18befe75013f7f6ec0815c72f"


def run(script, *arguments):
    print(f"Running {script.parent.name}/{script.name} {' '.join(arguments)}", flush=True)
    subprocess.run([sys.executable, "-X", "utf8", str(script), *arguments], cwd=script.parent, check=True)


def prepare_data():
    if not DATA.exists():
        listed = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", "-z", DATA_COMMIT, "--", "data/"], cwd=ROOT
        ).decode("utf-8")
        DATA.parent.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=DATA.parent) as temp:
            staged = Path(temp) / "data"
            staged.mkdir()
            for name in listed.split("\0"):
                member = Path(name)
                if member.parent == Path("data") and member.suffix == ".csv":
                    contents = subprocess.check_output(["git", "cat-file", "blob", f"{DATA_COMMIT}:{name}"], cwd=ROOT)
                    (staged / member.name).write_bytes(contents)
            run(V4 / "verify_data.py", "--data", str(staged))
            staged.rename(DATA)
        return
    run(V4 / "verify_data.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true", help="Rebuild daily models and selection artifacts before verifying")
    args = parser.parse_args()
    prepare_data()
    if args.retrain or not (V3 / "final_metrics.csv").exists():
        run(V3 / "reproduce.py")
    if args.retrain or not (HERE / "results" / "selection.json").exists():
        run(V4 / "run.py", "all")
        run(HERE / "transfer.py", "validate")
        run(HERE / "weather.py")
        run(HERE / "trees.py")
        run(HERE / "compare.py", "all")
    run(V4 / "audit_split.py")
    run(V4 / "intervals.py")
    run(V4 / "availability.py")
    run(V4 / "peak_audit.py")
    run(HERE / "uncertainty.py")
    run(HERE / "audit.py")
    run(HERE / "uncertainty.py", "--next")


if __name__ == "__main__":
    main()
