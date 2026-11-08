"""MaRS / Nevergrad Cobyla for high-J 2D context; fixed 1D Hamiltonian/widths.

--previous-fit REPORT inherits the complete search contract and best point.
Use --fit-config/--bounds/--initial/--fix/--vary; see README_2D.md.
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.fit_utils.fit_2d_context import main
else:
    from .fit_2d_context import main


if __name__ == "__main__":
    raise SystemExit(main(default_method="cobyla"))
