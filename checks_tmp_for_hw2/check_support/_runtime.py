"""Lazy numerical imports and Windows conda DLL discovery."""

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path


def numpy():
    if sys.platform == "win32":
        dll = Path(sys.prefix) / "Library" / "bin"
        if dll.is_dir() and str(dll) not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = str(dll) + os.pathsep + os.environ.get("PATH", "")
    return importlib.import_module("numpy")


@contextmanager
def numerical_threads(count):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=count, user_api="blas"):
        yield
