"""Convert space-separated DMRG table to semicolon-separated CSV."""
from pathlib import Path

SRC = Path("data/DMRG_table.csv")
DST = Path("data/DMRG_table_semicolon.csv")


def convert(src: Path, dst: Path) -> None:
    rows = [line.split() for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    dst.write_text("\n".join(";".join(r) for r in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    convert(SRC, DST)
    print(f"wrote {DST} ({DST.stat().st_size} bytes)")