from __future__ import annotations
import json, os, platform, subprocess, sys, time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

def _git(args: list[str]) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return None

@dataclass
class RunMeta:
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    git_sha: Optional[str] = field(default_factory=lambda: _git(["rev-parse", "HEAD"]))
    git_dirty: Optional[bool] = field(
        default_factory=lambda: bool(_git(["status", "--porcelain"]))
    )
    python: str = sys.version.split()[0]
    platform: str = platform.platform()
    cmd: str = " ".join(sys.argv)
    seed: Optional[int] = None
    env: Dict[str, Any] = field(default_factory=dict)
    packages: Dict[str, str] = field(default_factory=dict)
    note: Optional[str] = None

def write_meta(path: str, meta: RunMeta) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(meta), f, indent=2, sort_keys=True)