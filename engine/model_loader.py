from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def load_base_model(
    model_path,
    dtype=torch.float32,
    quantization=None,
    gradient_checkpointing=True,
):
    local = Path(model_path).exists()

    quantized = bool(
        quantization and quantization.get("enabled")
    )

    kwargs = {
        "dtype": (
            quantization["compute_dtype"]
            if quantized
            else dtype
        ),
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        "local_files_only": local,
    }

    if quantized:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Cannot load a quantized model without CUDA."
            )

        kwargs["quantization_config"] = quantization["config"]
        kwargs["device_map"] = {
            "": torch.cuda.current_device()
        }

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **kwargs,
    )

    model.config.use_cache = False

    if quantized:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            },
        )

    return model
