#!/usr/bin/env python3

import os
import re
import json
import shutil
import subprocess
from pathlib import Path


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def read_cpu_info():
    info = read_text("/proc/cpuinfo") or ""

    model = None
    for line in info.splitlines():
        if line.lower().startswith(("model name", "hardware")):
            model = line.split(":", 1)[-1].strip()
            break

    logical = os.cpu_count() or 1

    physical = None
    pairs = set()

    processor = None
    core_id = None
    physical_id = None

    for line in info.splitlines() + [""]:
        if not line.strip():
            if processor is not None:
                pairs.add((physical_id, core_id))
            processor = core_id = physical_id = None
            continue

        if ":" not in line:
            continue

        key, value = [part.strip() for part in line.split(":", 1)]

        if key == "processor":
            processor = value
        elif key == "core id":
            core_id = value
        elif key == "physical id":
            physical_id = value

    if pairs:
        physical = len(pairs)

    affinity = None
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass

    return {
        "model": model or "Unknown",
        "logical_cores": logical,
        "physical_cores": physical,
        "affinity_cores": affinity,
    }


def read_memory_info():
    values = {}
    info = read_text("/proc/meminfo") or ""

    for line in info.splitlines():
        match = re.match(r"^(\w+):\s+(\d+)\s+kB", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024

    total = values.get("MemTotal")
    available = values.get("MemAvailable")

    if available is None:
        available = values.get("MemFree", 0) + values.get("Buffers", 0)

    return {
        "total_bytes": total,
        "available_bytes": available,
    }


def read_cgroup_memory_limit():
    candidates = [
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ]

    for limit_path, usage_path in candidates:
        limit_text = read_text(limit_path)
        usage_text = read_text(usage_path)

        if not limit_text or limit_text == "max":
            continue

        try:
            limit = int(limit_text)
            usage = int(usage_text) if usage_text else None

            # Some cgroup v1 systems use a huge number for unlimited.
            if limit >= (1 << 60):
                continue

            return {
                "limit_bytes": limit,
                "usage_bytes": usage,
                "available_bytes": (
                    max(0, limit - usage) if usage is not None else None
                ),
            }
        except ValueError:
            continue

    return None


def read_nvidia_smi():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []

    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    devices = []

    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]

        if len(parts) < 4:
            continue

        try:
            devices.append({
                "name": parts[0],
                "memory_total_mib": int(parts[1]),
                "memory_free_mib": int(parts[2]),
                "driver_version": parts[3],
            })
        except ValueError:
            continue

    return devices


def detect_torch_cuda():
    try:
        import torch

        available = torch.cuda.is_available()
        devices = []

        if available:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)

                devices.append({
                    "index": index,
                    "name": props.name,
                    "memory_total_bytes": props.total_memory,
                    "bf16_supported": (
                        torch.cuda.is_bf16_supported()
                        if index == torch.cuda.current_device()
                        else None
                    ),
                })

        return {
            "available": available,
            "device_count": len(devices),
            "devices": devices,
        }

    except Exception as error:
        return {
            "available": False,
            "device_count": 0,
            "devices": [],
            "error": str(error),
        }


def detect_hardware():
    cpu = read_cpu_info()
    memory = read_memory_info()
    cgroup = read_cgroup_memory_limit()
    nvidia = read_nvidia_smi()
    cuda = detect_torch_cuda()

    effective_cores = cpu["affinity_cores"] or cpu["logical_cores"]

    # Respect container CPU quota when available.
    quota_cores = None
    quota = read_text("/sys/fs/cgroup/cpu.max")

    if quota:
        parts = quota.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota_cores = max(
                    1,
                    int(int(parts[0]) / int(parts[1]))
                )
            except (ValueError, ZeroDivisionError):
                pass

    if quota_cores is not None:
        effective_cores = min(effective_cores, quota_cores)

    recommended_threads = max(1, min(effective_cores, 4))

    return {
        "cpu": cpu,
        "effective_cpu_cores": effective_cores,
        "recommended_cpu_threads": recommended_threads,
        "memory": memory,
        "cgroup_memory": cgroup,
        "nvidia_smi": nvidia,
        "cuda": cuda,
    }


def format_gib(value):
    if value is None:
        return "Unknown"
    return f"{value / (1024 ** 3):.2f} GiB"


def print_hardware_report(report):
    cpu = report["cpu"]
    memory = report["memory"]
    cgroup = report["cgroup_memory"]
    cuda = report["cuda"]

    print("[UTE] ===== HARDWARE PROFILE =====", flush=True)
    print(f"[UTE] CPU: {cpu['model']}", flush=True)
    print(
        f"[UTE] CPU cores: physical={cpu['physical_cores']}, "
        f"logical={cpu['logical_cores']}, "
        f"affinity={cpu['affinity_cores']}",
        flush=True,
    )
    print(
        f"[UTE] Effective CPU cores: {report['effective_cpu_cores']}",
        flush=True,
    )
    print(
        f"[UTE] Recommended CPU threads: "
        f"{report['recommended_cpu_threads']}",
        flush=True,
    )
    print(
        f"[UTE] RAM total: {format_gib(memory['total_bytes'])}",
        flush=True,
    )
    print(
        f"[UTE] RAM available: {format_gib(memory['available_bytes'])}",
        flush=True,
    )

    if cgroup:
        print(
            f"[UTE] Container memory limit: "
            f"{format_gib(cgroup['limit_bytes'])}",
            flush=True,
        )
        print(
            f"[UTE] Container memory available: "
            f"{format_gib(cgroup['available_bytes'])}",
            flush=True,
        )
    else:
        print("[UTE] Container memory limit: Not detected", flush=True)

    print(f"[UTE] CUDA available: {cuda['available']}", flush=True)

    for device in cuda["devices"]:
        print(
            f"[UTE] GPU {device['index']}: {device['name']} "
            f"({format_gib(device['memory_total_bytes'])})",
            flush=True,
        )

    for device in report["nvidia_smi"]:
        print(
            f"[UTE] NVIDIA driver: {device['driver_version']}, "
            f"VRAM free: {device['memory_free_mib']} MiB",
            flush=True,
        )

    print("[UTE] ===== END HARDWARE PROFILE =====", flush=True)


def save_hardware_report(report, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    Path(path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
