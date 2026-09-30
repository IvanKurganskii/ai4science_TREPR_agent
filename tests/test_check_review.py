"""Exercise the review checker with valid and intentionally damaged submissions."""

from pathlib import Path
from shutil import copyfile
from subprocess import run
from sys import executable
from tempfile import TemporaryDirectory

from checks.check_review import PROJECT_DIR, TABLE_FILES, validate


def submission(directory: Path) -> Path:
    for filename in (*TABLE_FILES, "statistics.csv"):
        copyfile(PROJECT_DIR / "checks" / filename, directory / filename)
    return directory


def test_complete_submission_passes():
    with TemporaryDirectory(dir=PROJECT_DIR) as temp:
        directory = submission(Path(temp))
        assert validate(directory) == []
        result = run([executable, str(PROJECT_DIR / "checks" / "check_review.py"),
                      str(directory)], cwd=PROJECT_DIR, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_statistic_aliases_and_unicode_minus_pass():
    with TemporaryDirectory(dir=PROJECT_DIR) as temp:
        directory = submission(Path(temp))
        stats = directory / "statistics.csv"
        text = stats.read_text(encoding="utf-8")
        stats.write_text(text.replace("\nD,", "\n|D|,")
                        .replace("\nE,", "\n|E|,")
                        .replace("\neta,", "\n|E/D|,"), encoding="utf-8")
        table = directory / "Table_1.csv"
        table.write_text(table.read_text(encoding="utf-8").replace("-0.0118", "−0.0118"),
                         encoding="utf-8")
        assert validate(directory) == []


def test_missing_third_table_fails():
    with TemporaryDirectory(dir=PROJECT_DIR) as temp:
        directory = submission(Path(temp))
        (directory / "Table_3.csv").unlink()
        assert any("Table_3.csv" in error for error in validate(directory))


def test_changed_table_value_fails():
    with TemporaryDirectory(dir=PROJECT_DIR) as temp:
        directory = submission(Path(temp))
        path = directory / "Table_2.csv"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("0.0634,0.0642", "0.0635,0.0642"), encoding="utf-8")
        assert any("M_1024" in error for error in validate(directory))


def test_changed_statistic_fails():
    with TemporaryDirectory(dir=PROJECT_DIR) as temp:
        directory = submission(Path(temp))
        path = directory / "statistics.csv"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("65.466080794206", "68.0"), encoding="utf-8")
        assert any("eta" in error and "MAPE" in error for error in validate(directory))


def test_bound_must_remain_a_bound():
    with TemporaryDirectory(dir=PROJECT_DIR) as temp:
        directory = submission(Path(temp))
        path = directory / "Table_2.csv"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("<0.0001", "0.0001"), encoding="utf-8")
        assert any("M_2048" in error for error in validate(directory))


def test_method_comment_is_required():
    with TemporaryDirectory(dir=PROJECT_DIR) as temp:
        directory = submission(Path(temp))
        path = directory / "Table_1.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(line for line in lines if not line.startswith("#")) + "\n",
                        encoding="utf-8")
        assert any("method" in error for error in validate(directory))
