"""Run all experiments from the eight group CSVs in one fresh output directory."""
import argparse
from pathlib import Path
import subprocess
import sys

from scale_routing import ROOT, DATA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--out",type=Path,default=ROOT)
    args=parser.parse_args()
    data=args.data.resolve()
    out=args.out.resolve()
    steps=[
        ("run.py",["validate","--data",data,"--out",out]),
        ("feedback.py",["validate","--data",data,"--out",out]),
        ("run.py",["evaluate","--data",data,"--out",out]),
        ("feedback.py",["evaluate","--out",out]),
        ("audit_inputs.py",["--data",data,"--policy",out/"experiments","--routing-lock",out/"routing_results/lock.json"]),
        ("review.py",["--data",data,"--out",out]),
    ]
    for script,arguments in steps:
        subprocess.run([sys.executable,"-X","utf8",str(ROOT/script),*map(str,arguments)],check=True)


if __name__=="__main__":
    main()
