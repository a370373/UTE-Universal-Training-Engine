#!/usr/bin/env python3

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from engine.hardware import detect_hardware, print_hardware_report
from engine.preflight import run_preflight
from engine.smoke_test import run_smoke_test
from engine.trainer_integration_test import run_trainer_integration_test
from datetime import datetime, timezone

import torch
import transformers
import datasets
import peft
import trl

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


from engine.version import ENGINE_NAME, ENGINE_VERSION
from engine.adapter_verifier import verify_adapter_output

ALLOWED_ROLES = {"system", "user", "assistant"}


def log(message):
    print(f"[UTE] {message}", flush=True)


def fail(message):
    log(f"ERROR: {message}")
    raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal Hugging Face Causal LM SFT + LoRA Training Engine"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Local Base Model directory or supported Hugging Face model ID"
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Dataset root containing train/ and optional validation/"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="New output directory for this training run"
    )

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="all-linear",
        help="PEFT target modules: all-linear or comma-separated module names"
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume from a checkpoint inside the output directory"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate environment, model, tokenizer and dataset without training"
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Load the model and test LoRA injection without training"
    )

    parser.add_argument(
        "--preflight-report",
        default="reports/preflight.json",
        help="Path for the Preflight JSON report"
    )


    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one real LoRA forward/backward/optimizer step without training"
    )

    parser.add_argument(
        "--smoke-test-report",
        default="reports/smoke_test.json",
        help="Path for the Smoke Test JSON report"
    )

    parser.add_argument(
        "--trainer-integration-test",
        action="store_true",
        help="Run an isolated SFTTrainer training, save and reload integration test"
    )

    parser.add_argument(
        "--trainer-integration-report",
        default="reports/trainer_integration_test.json",
        help="Path for the Trainer Integration Test JSON report"
    )

    return parser.parse_args()


def check_environment(args):
    log(f"{ENGINE_NAME} v{ENGINE_VERSION}")
    log("Checking environment...")

    if args.cpu_threads < 1:
        fail("--cpu-threads must be >= 1")
    if args.max_length < 8:
        fail("--max-length must be >= 8")
    if args.epochs <= 0 or args.learning_rate <= 0:
        fail("Epochs and learning rate must be positive")
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        fail("Batch size and gradient accumulation must be >= 1")
    if args.lora_r < 1 or args.lora_alpha < 1:
        fail("LoRA rank and alpha must be >= 1")

    report = detect_hardware()
    print_hardware_report(report)

    effective_cores = max(1, report["effective_cpu_cores"])
    actual_threads = max(1, min(args.cpu_threads, effective_cores))

    torch.set_num_threads(actual_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cuda = report["cuda"]["available"]

    if cuda:
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            precision = "bf16"
        else:
            dtype = torch.float16
            precision = "fp16"
    else:
        dtype = torch.float32
        precision = "fp32"

    log(f"Python: {sys.version.split()[0]}")
    log(f"PyTorch: {torch.__version__}")
    log(f"Transformers: {transformers.__version__}")
    log(f"Datasets: {datasets.__version__}")
    log(f"PEFT: {peft.__version__}")
    log(f"TRL: {trl.__version__}")
    log(f"Device: {'CUDA' if cuda else 'CPU'}")
    log(f"Precision: {precision}")
    log(f"CPU threads selected: {actual_threads}")
    log("Hardware profiling completed.")

    return dtype, precision, cuda

def inspect_model(model_path):
    log(f"Inspecting Base Model: {model_path}")

    try:
        config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=Path(model_path).exists(),
            trust_remote_code=False
        )
    except Exception as error:
        fail(f"Cannot load model config: {error}")

    model_type = getattr(config, "model_type", None)
    architectures = getattr(config, "architectures", None)

    if not model_type:
        fail("Model config does not define model_type")

    if not architectures:
        log("WARNING: Model config does not declare architectures.")

    log(f"Model type: {model_type}")
    log(f"Architectures: {architectures}")
    log(f"Hidden size: {getattr(config, 'hidden_size', 'unknown')}")
    log(f"Layers: {getattr(config, 'num_hidden_layers', 'unknown')}")
    log(f"Vocabulary: {getattr(config, 'vocab_size', 'unknown')}")

    return config


def load_tokenizer(model_path):
    log("Loading tokenizer...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=Path(model_path).exists(),
            trust_remote_code=False
        )
    except Exception as error:
        fail(f"Tokenizer loading failed: {error}")

    if not tokenizer.chat_template:
        fail(
            "Tokenizer has no chat_template. "
            "A compatible conversation template must be configured first."
        )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            fail("Tokenizer has neither PAD nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    log(f"Tokenizer: {tokenizer.__class__.__name__}")
    log(f"Vocabulary size: {len(tokenizer)}")
    log("Chat template detected.")
    return tokenizer


def validate_record(item, source, line_number):
    if not isinstance(item, dict):
        raise ValueError("Record must be a JSON object")

    messages = item.get("messages")

    if not isinstance(messages, list) or not messages:
        raise ValueError("Missing or empty messages array")

    has_user = False
    has_assistant = False

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object")

        role = message.get("role")
        content = message.get("content")

        if role not in ALLOWED_ROLES:
            raise ValueError(f"Unsupported role: {role}")

        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message content must be non-empty text")

        has_user |= role == "user"
        has_assistant |= role == "assistant"

    if not has_user or not has_assistant:
        raise ValueError("Conversation needs user and assistant messages")

    return {"messages": messages}


def load_split(directory, required):
    directory = Path(directory)

    if not directory.is_dir():
        if required:
            fail(f"Required dataset directory not found: {directory}")
        log(f"Optional dataset directory not found: {directory}")
        return None

    files = sorted(directory.rglob("*.jsonl"))

    if not files:
        if required:
            fail(f"No JSONL files found in {directory}")
        log(f"No validation JSONL files found in {directory}")
        return None

    samples = []

    for filename in files:
        log(f"Reading {filename}")

        with filename.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                    samples.append(
                        validate_record(item, filename, line_number)
                    )
                except Exception as error:
                    fail(
                        f"Invalid dataset record at "
                        f"{filename}:{line_number}: {error}"
                    )

    if not samples:
        fail(f"No valid conversations found in {directory}")

    log(f"Loaded {len(samples)} conversation(s) from {directory}")
    return Dataset.from_list(samples)


def prepare_datasets(data_dir):
    data_dir = Path(data_dir)

    train_dataset = load_split(data_dir / "train", required=True)
    validation_dataset = load_split(
        data_dir / "validation",
        required=False
    )

    # The test directory is intentionally never loaded by the trainer.
    log("Test dataset is isolated and will not be used for training.")

    return train_dataset, validation_dataset


def prepare_output_directory(output_dir, resume_checkpoint):
    output_dir = Path(output_dir).resolve()

    if resume_checkpoint:
        checkpoint = Path(resume_checkpoint).resolve()

        if not output_dir.is_dir():
            fail("Output directory must already exist when resuming.")

        if output_dir not in checkpoint.parents:
            fail("Resume checkpoint must be inside the output directory.")

        if not checkpoint.is_dir():
            fail(f"Checkpoint directory not found: {checkpoint}")

        if not (checkpoint / "trainer_state.json").is_file():
            fail("Checkpoint is missing trainer_state.json.")

        return output_dir, str(checkpoint)

    if output_dir.exists() and any(output_dir.iterdir()):
        fail(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new directory to protect existing files."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, None


def build_model(model_path, dtype, cuda):
    log("Loading Base Model...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            local_files_only=Path(model_path).exists()
        )
    except Exception as error:
        fail(f"Base Model loading failed: {error}")

    model.config.use_cache = False

    if not cuda:
        log("CPU training selected. Large models may train slowly.")

    return model


def build_lora_config(args):
    if args.target_modules == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = [
            name.strip()
            for name in args.target_modules.split(",")
            if name.strip()
        ]

        if not target_modules:
            fail("No valid LoRA target modules were provided.")

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )


def write_run_metadata(
    output_dir,
    args,
    model_config,
    precision,
    train_dataset,
    validation_dataset
):
    metadata = {
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_type": getattr(model_config, "model_type", None),
        "architectures": getattr(model_config, "architectures", None),
        "precision": precision,
        "dataset": {
            "train_count": len(train_dataset),
            "validation_count": (
                len(validation_dataset)
                if validation_dataset is not None else 0
            ),
            "test_loaded": False
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules,
            "seed": args.seed
        }
    }

    path = output_dir / "run_config.json"
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    log(f"Run metadata written: {path}")


def main():
    args = parse_args()

    dtype, precision, cuda = check_environment(args)

    model_config = inspect_model(args.model)
    tokenizer = load_tokenizer(args.model)

    if args.preflight:
        lora_config = build_lora_config(args)

        report = run_preflight(
            args.model,
            lora_config,
            dtype,
            "cuda" if cuda else "cpu",
            args.preflight_report,
        )

        if not report["passed"]:
            fail("Preflight failed. Inspect the JSON report.")

        log("Preflight completed successfully.")
        return

    if args.smoke_test:
        log("===== TRAINING SMOKE TEST START =====")

        lora_config = build_lora_config(args)

        report = run_smoke_test(
            args.model,
            lora_config,
            dtype,
            args.max_length,
            args.cpu_threads,
            args.smoke_test_report,
        )

        if report["status"] != "passed":
            fail(
                "Smoke Test failed. "
                f"Inspect report: {args.smoke_test_report}"
            )

        log(f"Smoke Test report: {args.smoke_test_report}")
        log("TRAINING SMOKE TEST: PASS")
        log("===== TRAINING SMOKE TEST END =====")
        return

    if args.trainer_integration_test:
        log("===== TRAINER INTEGRATION TEST START =====")

        lora_config = build_lora_config(args)

        report = run_trainer_integration_test(
            args.model,
            lora_config,
            dtype,
            args.max_length,
            args.cpu_threads,
            cuda,
            args.trainer_integration_report,
        )

        log(f"Trainer Integration report: {args.trainer_integration_report}")

        if report["status"] != "passed":
            fail(
                "Trainer Integration Test failed. "
                f"Failed stage: {report.get('failed_stage', 'unknown')}. "
                f"Inspect report: {args.trainer_integration_report}"
            )

        log("TRAINER INTEGRATION TEST: PASS")
        log("===== TRAINER INTEGRATION TEST END =====")
        return

    train_dataset, validation_dataset = prepare_datasets(
        args.data_dir
    )

    if args.validate_only:
        log("Validation completed successfully.")
        log("No model weights were loaded and no training was started.")
        return

    output_dir, resume_checkpoint = prepare_output_directory(
        args.output_dir,
        args.resume_checkpoint
    )

    model = build_model(args.model, dtype, cuda)
    lora_config = build_lora_config(args)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy=(
            "epoch" if validation_dataset is not None else "no"
        ),
        use_cpu=not cuda,
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        optim="adamw_torch",
        max_length=args.max_length,
        packing=False,
        completion_only_loss=False,
        assistant_only_loss=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=True
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=lora_config
    )

    write_run_metadata(
        output_dir,
        args,
        model_config,
        precision,
        train_dataset,
        validation_dataset
    )

    log("Starting training...")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    log("Saving final adapter...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Release training memory before independent adapter verification.
    import gc

    del trainer
    del model
    gc.collect()

    if cuda:
        torch.cuda.empty_cache()

    log("Starting Adapter Output Verification...")

    verification_report = output_dir / "adapter_verification.json"

    verification = verify_adapter_output(
        model_path=args.model,
        adapter_dir=output_dir,
        dtype=dtype,
        max_length=args.max_length,
        report_path=verification_report,
    )

    if verification["status"] != "passed":
        fail(
            "Adapter Output Verification failed. "
            f"Stage: {verification.get('failed_stage')}. "
            f"Report: {verification_report}"
        )

    log("ADAPTER OUTPUT VERIFICATION: PASS")
    log(f"Verification report: {verification_report}")
    log(f"Training completed: {output_dir}")


if __name__ == "__main__":
    main()
