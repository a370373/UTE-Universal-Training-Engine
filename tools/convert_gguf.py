#!/usr/bin/env python3
"""
UTE GGUF Conversion Tool

Converts a Hugging Face model to GGUF using llama.cpp.

Optional quantization is performed using llama-quantize.
The original Hugging Face model is never modified.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


QUANT_TYPES = [
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q8_0",
    "Q2_K",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ3_XXS",
    "IQ3_S",
    "IQ3_M",
    "IQ1_S",
    "IQ1_M",
    "TQ1_0",
    "TQ2_0",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face model into GGUF format."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output GGUF file path.",
    )
    parser.add_argument(
        "--llama-cpp",
        required=True,
        help="Path to the llama.cpp repository.",
    )
    parser.add_argument(
        "--outtype",
        choices=["f32", "f16", "bf16"],
        default="f16",
        help="Intermediate GGUF precision. Default: f16.",
    )
    parser.add_argument(
        "--quantization",
        choices=QUANT_TYPES,
        default=None,
        help="Optional llama.cpp quantization type.",
    )
    parser.add_argument(
        "--quantize-bin",
        default=None,
        help="Optional path to llama-quantize executable.",
    )
    parser.add_argument(
        "--keep-f16",
        action="store_true",
        help="Keep the intermediate GGUF after quantization.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for conversion.",
    )

    return parser.parse_args()


def find_quantize_binary(llama_cpp, explicit_path=None):
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"llama-quantize not found: {path}"
            )

        return path

    candidates = [
        llama_cpp / "build" / "bin" / "llama-quantize",
        llama_cpp / "build" / "bin" / "Release" / "llama-quantize",
        llama_cpp / "bin" / "llama-quantize",
        llama_cpp / "llama-quantize",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    executable = shutil.which("llama-quantize")

    if executable:
        return Path(executable)

    raise FileNotFoundError(
        "llama-quantize was not found. Build llama.cpp or provide "
        "--quantize-bin."
    )


def run_command(command, description):
    print(f"\n{description}")
    print("-" * 68)
    print("Command:", " ".join(str(part) for part in command))
    print("-" * 68)

    subprocess.run(command, check=True)


def main():
    args = parse_args()

    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    llama_cpp = Path(args.llama_cpp).expanduser().resolve()

    if not model_path.is_dir():
        sys.exit(f"ERROR: Model directory does not exist: {model_path}")

    if not (model_path / "config.json").is_file():
        sys.exit("ERROR: config.json not found in model directory.")

    if not llama_cpp.is_dir():
        sys.exit(f"ERROR: llama.cpp directory does not exist: {llama_cpp}")

    converter = llama_cpp / "convert_hf_to_gguf.py"

    if not converter.is_file():
        sys.exit(
            "ERROR: convert_hf_to_gguf.py not found in:\n"
            f"{llama_cpp}"
        )

    if output_path.suffix.lower() != ".gguf":
        sys.exit("ERROR: Output filename must end with .gguf")

    if output_path.exists():
        sys.exit(
            f"ERROR: Output file already exists: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("UTE GGUF Conversion Tool")
    print("=" * 68)
    print(f"Model        : {model_path}")
    print(f"llama.cpp    : {llama_cpp}")
    print(f"Converter    : {converter}")
    print(f"Output       : {output_path}")
    print(f"Output type  : {args.outtype}")
    print(f"Quantization : {args.quantization or 'None'}")
    print("=" * 68)

    intermediate = None

    if args.quantization:
        intermediate = output_path.with_name(
            output_path.stem + ".intermediate.gguf"
        )

        if intermediate.exists():
            sys.exit(
                f"ERROR: Intermediate file already exists: {intermediate}"
            )

        conversion_output = intermediate
    else:
        conversion_output = output_path

    try:
        conversion_command = [
            args.python,
            str(converter),
            str(model_path),
            "--outfile",
            str(conversion_output),
            "--outtype",
            args.outtype,
        ]

        run_command(
            conversion_command,
            "[1/2] Converting Hugging Face model to GGUF...",
        )

        if not conversion_output.is_file():
            sys.exit(
                f"ERROR: Converter did not create output: {conversion_output}"
            )

        if args.quantization:
            quantize_binary = find_quantize_binary(
                llama_cpp,
                args.quantize_bin,
            )

            quantization_command = [
                str(quantize_binary),
                str(intermediate),
                str(output_path),
                args.quantization,
            ]

            run_command(
                quantization_command,
                "[2/2] Quantizing GGUF model...",
            )

            if not output_path.is_file():
                sys.exit(
                    "ERROR: Quantizer did not create the output file."
                )

            if not args.keep_f16:
                intermediate.unlink()
                print("Intermediate GGUF removed.")

            else:
                print(f"Intermediate GGUF preserved: {intermediate}")

        print("\n" + "=" * 68)
        print("GGUF CONVERSION COMPLETED")
        print("=" * 68)
        print(f"Output: {output_path}")
        print(f"Size  : {output_path.stat().st_size:,} bytes")

    except subprocess.CalledProcessError as exc:
        print(f"\nERROR: External command failed with exit code {exc.returncode}")
        if intermediate and intermediate.exists():
            print(f"Intermediate file preserved: {intermediate}")
        sys.exit(exc.returncode)

    except (OSError, FileNotFoundError) as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
