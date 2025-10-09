from __future__ import annotations
import shutil, time, pathlib
from typing import Union

def backup_file(p: Union[str, pathlib.Path]) -> None:
    p = pathlib.Path(p)
    if not p.exists():  # nothing to back up
        return
    ts = time.strftime("%Y%m%dT%H%M%S")
    backup_path = p.with_name(f"{p.name}.backup.{ts}")
    shutil.copy2(p, backup_path)