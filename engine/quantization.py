import importlib.util

import torch


def resolve_quantization(args, cuda, dtype, precision):
    mode = args.quantization

    if mode == "none":
        return {
            "enabled": False,
            "mode": "none",
            "config": None,
            "compute_dtype": dtype,
            "precision": precision,
            "metadata": {
                "method": "LoRA",
                "quantization": "none",
            },
        }

    if mode != "4bit":
        raise ValueError(f"Unsupported quantization mode: {mode}")

    if not cuda or not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA 4-bit currently requires a supported CUDA GPU."
        )

    if not importlib.util.find_spec("bitsandbytes"):
        raise RuntimeError(
            "QLoRA requires bitsandbytes. "
            "Install requirements-qlora.txt first."
        )

    if not importlib.util.find_spec("accelerate"):
        raise RuntimeError(
            "QLoRA requires Accelerate. "
            "Install requirements-qlora.txt first."
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "The initial QLoRA implementation supports one CUDA GPU. "
            "Multi-GPU quantized training is not enabled yet."
        )

    requested_dtype = args.compute_dtype

    if requested_dtype == "auto":
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    elif requested_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "BF16 was requested, but this CUDA GPU does not "
                "report BF16 support."
            )
        compute_dtype = torch.bfloat16
    elif requested_dtype == "fp16":
        compute_dtype = torch.float16
    else:
        raise ValueError(
            f"Unsupported compute dtype: {requested_dtype}"
        )

    compute_precision = (
        "bf16"
        if compute_dtype == torch.bfloat16
        else "fp16"
    )

    try:
        from transformers import BitsAndBytesConfig

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.quant_type,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    except Exception as error:
        raise RuntimeError(
            f"Cannot create BitsAndBytesConfig: {error}"
        ) from error

    return {
        "enabled": True,
        "mode": "4bit",
        "config": config,
        "compute_dtype": compute_dtype,
        "precision": compute_precision,
        "metadata": {
            "method": "QLoRA",
            "quantization": "4bit",
            "quant_type": args.quant_type,
            "double_quant": args.double_quant,
            "compute_dtype": compute_precision,
            "device": "cuda:0",
            "gpu_count": 1,
        },
    }
