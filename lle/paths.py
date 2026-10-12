"""Where everything lives. A layout change is made here and nowhere else."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GC = ROOT / "data" / "gc"  # raw .gcd, one folder per GC batch
WORKBOOKS = ROOT / "data" / "workbooks"  # hand-made inputs
ENTRY = ROOT / "data" / "entry"  # typed repeat-round entry sheets (lab record)
TEMPLATES = ROOT / "templates"
OUT = ROOT / "out"  # regenerable only
PCSAFT_QUATERNARY = ROOT / "pcsaft-quaternary"
