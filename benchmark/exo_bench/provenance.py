from __future__ import annotations

import platform
import subprocess

import mujoco
import numpy as np
import torch

from .hashing import sha256_file, sha256_tree
from .paths import BENCHMARK_ROOT, REPOSITORY_ROOT, UV_LOCK_PATH


def git_provenance() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_output = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(dirty_output.strip()),
        "source_tree_sha256": sha256_tree(
            BENCHMARK_ROOT / "exo_bench",
            include=lambda path: path.suffix == ".py",
        ),
    }


def runtime_provenance() -> dict:
    return {
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "dependency_lock": {
            "path": "uv.lock",
            "sha256": sha256_file(UV_LOCK_PATH),
        },
    }


def manifest_hashes(robot_manifest_path, controller_manifest_path) -> dict:
    return {
        "robot_manifest_sha256": sha256_file(robot_manifest_path),
        "controller_manifest_sha256": sha256_file(controller_manifest_path),
    }
