"""paths — resolve the raw-data roots by **searching for signature files**.

Why this is needed
------------------
On 2026-09-10 the contents of `<DATA_ROOT_ALT>` were copied into `rawfiles_fromJY/DATA/Data/`, which changed the layout.
The old curated structure (`DATA/Data`, `DATA/IIT_data`, `DATA/VIP_all`, `DATA/Raw_data/EC`) and
the new E: mirror structure (`DATA/Data/Data`, `DATA/Data/iit_delivery/Data`, `DATA/Data/Data/VIP_all`,
`DATA/Data/Raw_data/EC`) **differ even in folder names** — it is not a simple one-level move.

So paths are not hard-coded; each root is found by **a file that must exist inside it (its signature)**.
If the data are moved again, the code follows.

**Read-only**: this module writes nothing.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

if not __package__: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script: put code/shared on sys.path
import local_paths as LP  # noqa: E402

# Search starting points (earlier entries take precedence)
BASES = [
    LP.optional("DATA_ROOT", "DATA"),
    LP.optional("DATA_ROOT"),
    LP.optional("DATA_ROOT_ALT"),
]
BASES = [b for b in BASES if b is not None]  # roots not set in local_paths are skipped
MAX_DEPTH = 5


def _walk(base: Path, depth: int):
    """Breadth-first walk over the directories up to `depth` levels below `base`."""
    yield base
    frontier = [base]
    for _ in range(depth):
        nxt = []
        for d in frontier:
            try:
                for c in d.iterdir():
                    if c.is_dir():
                        yield c
                        nxt.append(c)
            except (PermissionError, OSError):
                continue
        frontier = nxt


@lru_cache(maxsize=64)
def find_dir(*signature: str) -> Path | None:
    """Return the first directory that contains **all** of the `signature` relative paths."""
    for base in BASES:
        if not base.exists():
            continue
        for d in _walk(base, MAX_DEPTH):
            if all((d / s).exists() for s in signature):
                return d
    return None


@lru_cache(maxsize=64)
def find_file(name: str, *must_contain: str) -> Path | None:
    """Find a file named `name`. `must_contain` are fragments that must appear in the path string."""
    for base in BASES:
        if not base.exists():
            continue
        for p in base.rglob(name):
            sp = str(p).replace("\\", "/")
            if all(m in sp for m in must_contain):
                return p
    return None


# ── Per-cohort roots (by signature file) ─────────────────────
def data_root() -> Path | None:
    """Folder holding Recording_metadata.xlsx + TB/ (formerly `DATA/Data`)."""
    return find_dir("Recording_metadata.xlsx", "TB")


def jy_source_root() -> Path | None:
    """Original thesis-pipeline code folder (contains utils/eeg_utils/SO_Spindle_coupling.py)."""
    return find_dir("utils/eeg_utils/SO_Spindle_coupling.py")


def iit_root() -> Path | None:
    """Folder holding IIT_Recording_metadata.xlsx."""
    return find_dir("IIT_Recording_metadata.xlsx")


def vip_all_root() -> Path | None:
    """Folder holding VIPALL_Recording_metadata.xlsx."""
    return find_dir("VIPALL_Recording_metadata.xlsx")


def ec_raw_root() -> Path | None:
    """EC raw-CSV root — a folder where CSVs actually exist for both active (A_) and sham (B_)."""
    def arms(root: Path) -> set[str]:
        return {d.name[0] for d in root.glob("[AB]_*")
                if d.is_dir() and any(d.glob("*.csv"))}
    cands = []
    for base in BASES:
        if not base.exists():
            continue
        for d in _walk(base, MAX_DEPTH):
            try:
                a = arms(d)
            except (PermissionError, OSError):
                continue
            if a:
                cands.append((d, a))
    for d, a in cands:              # prefer folders that have both active+sham
        if a >= {"A", "B"}:
            return d
    return cands[0][0] if cands else None


def report() -> dict[str, Path | None]:
    out = {
        "data_root (Recording_metadata + TB)": data_root(),
        "jy_source_root (원본 검출기 코드)": jy_source_root(),
        "iit_root": iit_root(),
        "vip_all_root": vip_all_root(),
        "ec_raw_root": ec_raw_root(),
    }
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("원본 데이터 루트 해석 결과")
    for k, v in report().items():
        print(f"  {k:38s} {v if v else '  ** 찾지 못함 **'}")
