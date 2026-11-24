import json
from datetime import datetime, timezone
from pathlib import Path

from engine.compatibility import inspect_compatibility
from engine.lora_inspector import inspect_lora_injection
from engine.version import ENGINE_NAME, ENGINE_VERSION


def run_preflight(
    model_path,
    lora_config,
    dtype,
    device,
    report_path,
):
    print("[UTE] ===== PREFLIGHT START =====", flush=True)

    compatibility = inspect_compatibility(model_path)

    print(
        "[UTE] Model compatibility:",
        "PASS" if compatibility["compatible"] else "FAIL",
        flush=True,
    )

    lora_report = {
        "compatible": False,
        "errors": ["Skipped because model compatibility failed."],
    }

    if compatibility["compatible"]:
        print("[UTE] Testing LoRA injection...", flush=True)

        lora_report = inspect_lora_injection(
            model_path,
            lora_config,
            dtype,
            device,
        )

        print(
            "[UTE] LoRA injection:",
            "PASS" if lora_report["compatible"] else "FAIL",
            flush=True,
        )

    report = {
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "compatibility": compatibility,
        "lora": lora_report,
        "passed": (
            compatibility["compatible"]
            and lora_report["compatible"]
        ),
    }

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[UTE] Preflight report: {report_path}", flush=True)
    print(
        "[UTE] PREFLIGHT RESULT:",
        "PASS" if report["passed"] else "FAIL",
        flush=True,
    )
    print("[UTE] ===== PREFLIGHT END =====", flush=True)

    return report
