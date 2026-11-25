import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from peft import PeftModel
from safetensors import safe_open
from transformers import AutoTokenizer
from engine.model_loader import load_base_model

from engine.version import ENGINE_VERSION


def write_report(report, report_path):
    if report_path is None:
        return

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def verify_adapter_output(
    model_path,
    adapter_dir,
    dtype=torch.float32,
    max_length=128,
    report_path=None,
    quantization=None,
):
    adapter_dir = Path(adapter_dir)

    report = {
        "test": "adapter_output_verification",
        "engine_version": ENGINE_VERSION,
        "model_path": str(model_path),
        "adapter_dir": str(adapter_dir),
        "quantization": (
            quantization["metadata"]
            if quantization is not None
            else {"method": "LoRA", "quantization": "none"}
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "checks": {
            "config": False,
            "weights": False,
            "metadata": False,
            "reload": False,
            "forward": False,
        },
        "errors": [],
    }

    failed_stage = None
    base_model = None
    adapter_model = None

    try:
        # 1. Adapter configuration
        failed_stage = "config"

        config_path = adapter_dir / "adapter_config.json"

        if not config_path.is_file():
            raise FileNotFoundError("adapter_config.json is missing")

        config = json.loads(config_path.read_text(encoding="utf-8"))

        if not isinstance(config, dict):
            raise ValueError("Adapter configuration must be a JSON object")

        report["checks"]["config"] = True

        # 2. Safetensors weights
        failed_stage = "weights"

        weights_path = adapter_dir / "adapter_model.safetensors"

        if not weights_path.is_file():
            raise FileNotFoundError("adapter_model.safetensors is missing")

        if weights_path.stat().st_size <= 0:
            raise ValueError("Adapter weights file is empty")

        with safe_open(str(weights_path), framework="pt", device="cpu") as f:
            weight_keys = list(f.keys())

        if not weight_keys:
            raise ValueError("Adapter weights contain no tensors")

        report["weight_file"] = weights_path.name
        report["weight_size_bytes"] = weights_path.stat().st_size
        report["weight_tensor_count"] = len(weight_keys)

        report["checks"]["weights"] = True

        # 3. Metadata
        failed_stage = "metadata"

        peft_type = config.get("peft_type")
        rank = config.get("r")
        target_modules = config.get("target_modules")

        if peft_type != "LORA":
            raise ValueError(f"Expected LoRA adapter, got: {peft_type}")

        if not isinstance(rank, int) or rank <= 0:
            raise ValueError("Invalid LoRA rank in adapter configuration")

        if not target_modules:
            raise ValueError("Adapter target_modules are missing")

        report["metadata"] = {
            "peft_type": peft_type,
            "r": rank,
            "target_modules": target_modules,
            "base_model_name_or_path": config.get(
                "base_model_name_or_path"
            ),
        }

        report["checks"]["metadata"] = True

        # 4. Reload Base Model and Adapter
        failed_stage = "reload"

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False,
            local_files_only=Path(model_path).exists(),
        )

        base_model = load_base_model(
            model_path=model_path,
            dtype=dtype,
            quantization=quantization,
            gradient_checkpointing=False,
        )

        adapter_model = PeftModel.from_pretrained(
            base_model,
            str(adapter_dir),
            is_trainable=False,
        )

        report["checks"]["reload"] = True

        # 5. Forward test
        failed_stage = "forward"

        adapter_model.eval()

        inputs = tokenizer(
            "Adapter verification test.",
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        device = adapter_model.get_input_embeddings().weight.device

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = adapter_model(**inputs)

        logits = outputs.logits

        if logits.ndim != 3:
            raise ValueError("Unexpected model logits dimensions")

        if not torch.isfinite(logits).all().item():
            raise ValueError("Forward output contains NaN or Inf")

        report["forward"] = {
            "logits_shape": list(logits.shape),
            "finite": True,
        }

        report["checks"]["forward"] = True
        report["status"] = "passed"

    except Exception as error:
        report["errors"].append(
            f"{type(error).__name__}: {error}"
        )
        report["failed_stage"] = failed_stage
        report["traceback"] = traceback.format_exc()

    finally:
        adapter_model = None
        base_model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report["duration_note"] = (
        "Verification completed. "
        "No training or Base Model modification was performed."
    )

    write_report(report, report_path)

    return report
