#!/usr/bin/env python3
"""Fail-closed NVIDIA free-memory gate and lab-local training lock."""
from __future__ import annotations

import fcntl
import os
import subprocess
import time
from pathlib import Path


def acquire_training_lock(path: Path, timeout: float, interval: float) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise TimeoutError(f"Lab GPU training lock is busy: {path}")
            time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
        except BaseException:
            os.close(fd)
            raise


def free_memory_mib(gpu_index: int, executable: str = "nvidia-smi") -> int:
    result = subprocess.run(
        [executable, f"--id={gpu_index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    value = result.stdout.strip()
    if not value.isdigit():
        raise ValueError(f"Unexpected nvidia-smi free-memory result: {value!r}")
    return int(value)


def wait_for_vram(gpu_index: int, minimum_mib: int, timeout: float, interval: float,
                  executable: str = "nvidia-smi") -> int:
    if gpu_index < 0 or minimum_mib <= 0 or timeout < 0 or interval <= 0:
        raise ValueError("GPU index, minimum free MiB, timeout, or polling interval is invalid")
    deadline = time.monotonic() + timeout
    while True:
        free_mib = free_memory_mib(gpu_index, executable)
        if free_mib >= minimum_mib:
            return free_mib
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPU {gpu_index} has {free_mib} MiB free; need {minimum_mib} MiB")
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
