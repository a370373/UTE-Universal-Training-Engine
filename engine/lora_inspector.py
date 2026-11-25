import gc

import torch
from peft import get_peft_model

from engine.model_loader import load_base_model


def _is_supported_linear(module):
    if isinstance(module, torch.nn.Linear):
        return True

    # bitsandbytes quantized linear layers.
    module_name = module.__class__.__name__
    module_package = module.__class__.__module__

    return (
        module_package.startswith("bitsandbytes")
        and module_name in {"Linear4bit", "Linear8bitLt"}
    )


def inspect_lora_injection(
    model_path,
    lora_config,
    dtype,
    device,
    quantization=None,
):
    report = {
        "model_loaded": False,
        "injection_success": False,
        "target_modules": [],
        "total_parameters": 0,
        "trainable_parameters": 0,
        "trainable_percentage": 0.0,
        "embedding_shape": None,
        "quantization": (
            quantization["metadata"]
            if quantization is not None
            else {"method": "LoRA", "quantization": "none"}
        ),
        "errors": [],
    }

    model = None
    peft_model = None

    try:
        model = load_base_model(
            model_path=model_path,
            dtype=dtype,
            quantization=quantization,
            gradient_checkpointing=False,
        )

        report["model_loaded"] = True

        embeddings = model.get_input_embeddings()

        if embeddings is None:
            raise RuntimeError(
                "Model has no input embedding layer."
            )

        report["embedding_shape"] = list(
            embeddings.weight.shape
        )

        config_vocab = getattr(
            model.config, "vocab_size", None
        )
        embedding_vocab = embeddings.weight.shape[0]

        if (
            config_vocab is not None
            and config_vocab != embedding_vocab
        ):
            report["errors"].append(
                "Config vocabulary and embedding dimensions differ."
            )

        if embedding_vocab < 1:
            raise RuntimeError(
                "Invalid embedding dimensions."
            )

        linear_modules = [
            name
            for name, module in model.named_modules()
            if name and _is_supported_linear(module)
        ]

        if not linear_modules:
            raise RuntimeError(
                "No supported Linear modules detected."
            )

        requested = lora_config.target_modules

        if isinstance(requested, str):
            if requested != "all-linear":
                raise RuntimeError(
                    f"Unsupported target module selector: {requested}"
                )

            matched_modules = linear_modules

        else:
            matched_modules = [
                name
                for name in linear_modules
                if any(
                    name == target
                    or name.endswith("." + target)
                    for target in requested
                )
            ]

            missing = [
                target
                for target in requested
                if not any(
                    name == target
                    or name.endswith("." + target)
                    for name in linear_modules
                )
            ]

            if missing:
                raise RuntimeError(
                    f"Target modules not found: {missing}"
                )

        report["target_modules"] = sorted(
            set(matched_modules)
        )

        peft_model = get_peft_model(
            model,
            lora_config,
        )

        total = sum(
            parameter.numel()
            for parameter in peft_model.parameters()
        )

        trainable = sum(
            parameter.numel()
            for parameter in peft_model.parameters()
            if parameter.requires_grad
        )

        report["total_parameters"] = total
        report["trainable_parameters"] = trainable
        report["trainable_percentage"] = (
            trainable / total * 100 if total else 0.0
        )

        if trainable <= 0:
            raise RuntimeError(
                "LoRA injection produced no trainable parameters."
            )

        if trainable >= total:
            raise RuntimeError(
                "All model parameters are trainable; "
                "expected PEFT-only training."
            )

        report["injection_success"] = True

    except Exception as error:
        report["errors"].append(
            f"{type(error).__name__}: {error}"
        )

    finally:
        peft_model = None
        model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report["compatible"] = (
        report["model_loaded"]
        and report["injection_success"]
        and not report["errors"]
    )

    return report
