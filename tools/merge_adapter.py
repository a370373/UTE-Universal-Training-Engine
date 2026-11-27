#!/usr/bin/env python3
"""
UTE Model Merge Tool

Supports:
- Standard LoRA adapter merging
- QLoRA adapter merging against the original full-precision base model
- PEFT modules_to_save, including input embeddings and output LM head

The script deliberately loads the base model without 4-bit/8-bit
quantization. For QLoRA, provide the original Hugging Face base model.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA/QLoRA adapters into a full Hugging Face model."
    )

    parser.add_argument("--base-model", required=True,
                        help="Original base model directory or Hugging Face model ID.")
    parser.add_argument("--adapter", required=True,
                        help="PEFT adapter directory.")
    parser.add_argument("--output", required=True,
                        help="Output directory for the merged model.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Precision used to load the base model. Default: auto.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code from the model repository.",
    )

    return parser.parse_args()


def load_dtype(name):
    if name == "auto":
        return "auto"

    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def read_adapter_config(adapter_path):
    config_path = adapter_path / "adapter_config.json"

    if not config_path.is_file():
        sys.exit(
            f"ERROR: PEFT adapter_config.json not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"ERROR: Cannot read adapter config: {exc}")


def validate_paths(adapter_path, output_path):
    if not adapter_path.is_dir():
        sys.exit(f"ERROR: Adapter directory does not exist: {adapter_path}")

    if output_path == adapter_path:
        sys.exit("ERROR: Output directory cannot be the adapter directory.")

    if output_path.exists():
        if not output_path.is_dir():
            sys.exit("ERROR: Output path exists and is not a directory.")

        if any(output_path.iterdir()):
            sys.exit(
                "ERROR: Output directory is not empty. "
                "Choose a new output directory."
            )


def preserve_file(sources, output_path, filename):
    destination = output_path / filename

    if destination.exists():
        return

    for source_dir in sources:
        if source_dir is None:
            continue

        source = source_dir / filename

        if source.is_file():
            shutil.copy2(source, destination)
            print(f"Preserved: {filename}")
            return


def main():
    args = parse_args()

    base_model = args.base_model
    adapter_path = Path(args.adapter).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    validate_paths(adapter_path, output_path)

    adapter_config = read_adapter_config(adapter_path)

    peft_type = adapter_config.get("peft_type", "unknown")
    modules_to_save = adapter_config.get("modules_to_save") or []
    target_modules = adapter_config.get("target_modules") or []

    print("=" * 68)
    print("UTE Model Merge Tool")
    print("=" * 68)
    print(f"Base model      : {base_model}")
    print(f"Adapter         : {adapter_path}")
    print(f"Output          : {output_path}")
    print(f"PEFT type       : {peft_type}")
    print(f"Target modules  : {target_modules}")
    print(f"Modules to save : {modules_to_save}")
    print(f"Loading dtype   : {args.dtype}")
    print("=" * 68)

    if peft_type.upper() != "LORA":
        sys.exit(
            f"ERROR: This tool currently supports PEFT LORA adapters, "
            f"not {peft_type}."
        )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        sys.exit(
            f"ERROR: Missing dependency: {exc}\n"
            "Install with: pip install torch transformers peft safetensors"
        )

    # QLoRA merge strategy:
    # Load the ORIGINAL base model in a normal floating-point dtype.
    # Do not load it with load_in_4bit/load_in_8bit.
    print("\n[1/6] Loading tokenizer...")

    local_base_path = Path(base_model).expanduser()
    local_base_dir = local_base_path if local_base_path.is_dir() else None

    adapter_has_tokenizer = (
        (adapter_path / "tokenizer.json").is_file()
        or (adapter_path / "tokenizer_config.json").is_file()
    )

    tokenizer_source = (
        str(adapter_path) if adapter_has_tokenizer else base_model
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Tokenizer source: {tokenizer_source}")

    print("\n[2/6] Loading base model in floating-point precision...")
    print("Quantized loading is intentionally disabled for merging.")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=load_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    print(f"Base model loaded. dtype setting: {args.dtype}")

    print("\n[3/6] Loading PEFT adapter...")

    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=False,
    )

    print("\n[4/6] Merging adapter into base model...")

    merged_model = model.merge_and_unload(safe_merge=True)
    merged_model.eval()

    # Ensure the merge did not leave PEFT wrappers in the model.
    remaining_wrappers = [
        name for name, module in merged_model.named_modules()
        if module.__class__.__name__ == "ModulesToSaveWrapper"
    ]

    if remaining_wrappers:
        sys.exit(
            "ERROR: PEFT ModulesToSaveWrapper objects remain after merge: "
            + ", ".join(remaining_wrappers)
        )

    print("\n[5/6] Saving merged model...")

    output_path.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(
        str(output_path),
        safe_serialization=True,
    )

    tokenizer.save_pretrained(str(output_path))

    # Preserve external model files when available locally.
    preserve_file(
        [adapter_path, local_base_dir],
        output_path,
        "chat_template.jinja",
    )
    preserve_file(
        [adapter_path, local_base_dir],
        output_path,
        "generation_config.json",
    )

    print("\n[6/6] Checking output artifacts...")

    weight_files = list(output_path.glob("*.safetensors"))
    if not weight_files:
        sys.exit("ERROR: No safetensors model weights found in output.")

    if not (output_path / "config.json").is_file():
        sys.exit("ERROR: config.json was not saved.")

    print(f"Weight files: {len(weight_files)}")
    print(f"Output directory: {output_path}")

    print("\n" + "=" * 68)
    print("MERGE COMPLETED")
    print("=" * 68)
    print("The merged model is saved in Hugging Face format.")

    if modules_to_save:
        print("\nmodules_to_save configured in adapter:")
        for name in modules_to_save:
            print(f"  - {name}")
        print(
            "PEFT merge_and_unload() was used to merge these saved modules. "
            "Tensor-level verification is still recommended."
        )
    else:
        print("\nNo modules_to_save configured.")

    print("\nSupported workflow:")
    print("  - Standard LoRA adapter")
    print("  - QLoRA adapter + original floating-point base model")
    print("  - PEFT modules_to_save (e.g. embeddings / LM head)")


if __name__ == "__main__":
    main()
