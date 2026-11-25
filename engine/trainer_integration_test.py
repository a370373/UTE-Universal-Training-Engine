import gc
import json
import math
import tempfile
import time
import traceback

from datetime import datetime, timezone
from pathlib import Path

import torch

from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import PeftModel
from engine.model_loader import load_base_model
from engine.embedding_training import configure_embedding_training
from engine.version import ENGINE_VERSION


def run_trainer_integration_test(
    model_path,
    lora_config,
    dtype=torch.float32,
    max_length=128,
    cpu_threads=4,
    cuda=False,
    report_path=None,
    quantization=None,
    train_embeddings=False,
):
    started = time.time()

    report = {
        "test": "trainer_integration_test",
        "engine_version": ENGINE_VERSION,
        "model_path": str(model_path),
        "quantization": (
            quantization["metadata"]
            if quantization is not None
            else {"method": "LoRA", "quantization": "none"}
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "checks": {},
        "errors": [],
    }

    stage = "initialization"
    trainer = None
    model = None
    base_model = None
    reloaded_model = None
    tokenizer = None
    temporary_directory = None

    try:
        torch.set_num_threads(max(1, cpu_threads))

        model_path = Path(model_path)

        if not model_path.exists():
            raise RuntimeError(
                "Integration test currently requires a local model directory."
            )

        stage = "tokenizer_loading"

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=False,
            local_files_only=True,
        )

        if not tokenizer.chat_template:
            raise RuntimeError("Tokenizer has no chat template.")

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("Tokenizer has no PAD or EOS token.")
            tokenizer.pad_token = tokenizer.eos_token

        report["checks"]["tokenizer_loading"] = True

        stage = "dataset_creation"

        dataset = Dataset.from_list([
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "請簡單介紹人工智慧。"
                    },
                    {
                        "role": "assistant",
                        "content": "人工智慧是讓電腦執行需要智慧的任務的技術。"
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "什麼是 Python？"
                    },
                    {
                        "role": "assistant",
                        "content": "Python 是一種常用的程式語言。"
                    }
                ]
            },
        ])

        if len(dataset) != 2:
            raise RuntimeError("Unexpected integration dataset size.")

        report["checks"]["dataset_creation"] = True
        report["dataset_samples"] = len(dataset)

        stage = "temporary_output_creation"

        report_file = (
            Path(report_path).resolve()
            if report_path else
            Path("reports/trainer_integration_test.json").resolve()
        )

        report_file.parent.mkdir(parents=True, exist_ok=True)

        temporary_directory = tempfile.TemporaryDirectory(
            prefix="ute_trainer_test_",
            dir=str(report_file.parent),
        )

        temporary_root = Path(temporary_directory.name)
        trainer_output = temporary_root / "trainer_output"
        adapter_output = temporary_root / "adapter"

        stage = "model_loading"

        model = load_base_model(
            model_path=str(model_path),
            dtype=dtype,
            quantization=quantization,
            gradient_checkpointing=False,
        )

        report["checks"]["model_loading"] = True

        if train_embeddings:
            stage = "embedding_training_configuration"

            embedding_info = configure_embedding_training(
                model,
                lora_config
            )

            report["embedding_training"] = {
                "enabled": True,
                **embedding_info,
            }
        else:
            report["embedding_training"] = {
                "enabled": False,
            }


        stage = "trainer_configuration"

        training_args = SFTConfig(
            output_dir=str(trainer_output),
            max_steps=1,
            num_train_epochs=1,
            learning_rate=1e-4,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            logging_strategy="steps",
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no",
            optim="adamw_torch",
            use_cpu=not cuda,
            bf16=(
                quantization["precision"] == "bf16"
                if quantization is not None
                else False
            ),
            fp16=(
                quantization["precision"] == "fp16"
                if quantization is not None
                else False
            ),
            max_length=max(8, min(max_length, 128)),
            packing=False,
            completion_only_loss=False,
            assistant_only_loss=False,
            gradient_checkpointing=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            report_to="none",
            remove_unused_columns=True,
                seed=42,
        )

        report["checks"]["trainer_configuration"] = True

        stage = "trainer_initialization"

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )

        report["checks"]["trainer_initialization"] = True

        stage = "training"

        result = trainer.train()

        training_loss = float(result.training_loss)

        if not math.isfinite(training_loss):
            raise RuntimeError("Trainer produced a non-finite training loss.")

        report["training_loss"] = training_loss
        report["global_step"] = int(trainer.state.global_step)
        report["checks"]["training"] = True

        if trainer.state.global_step < 1:
            raise RuntimeError("Trainer did not complete an optimizer step.")

        stage = "adapter_save"

        trainer.save_model(str(adapter_output))
        tokenizer.save_pretrained(str(adapter_output))

        adapter_config = adapter_output / "adapter_config.json"

        adapter_weights = (
            adapter_output / "adapter_model.safetensors"
        )

        if not adapter_config.is_file():
            raise RuntimeError("Adapter config was not saved.")

        if not adapter_weights.is_file():
            adapter_weights = adapter_output / "adapter_model.bin"

        if train_embeddings:
            saved_config = json.loads(
                adapter_config.read_text(encoding="utf-8")
            )

            saved_modules = saved_config.get("modules_to_save") or []
            expected_modules = set(
                report["embedding_training"]["modules_to_save"]
            )

            if not expected_modules.issubset(set(saved_modules)):
                raise RuntimeError(
                    "Saved adapter config does not contain all expected "
                    "embedding modules. "
                    f"Expected: {sorted(expected_modules)}; "
                    f"saved: {saved_modules}"
                )

            report["checks"]["embedding_modules_saved"] = True

        if not adapter_weights.is_file():
            raise RuntimeError("Adapter weights were not saved.")

        report["checks"]["adapter_save"] = True

        report["adapter_files"] = [
            adapter_config.name,
            adapter_weights.name,
        ]

        stage = "release_training_resources"

        del trainer
        trainer = None

        del model
        model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stage = "adapter_reload"

        base_model = load_base_model(
            model_path=str(model_path),
            dtype=dtype,
            quantization=quantization,
            gradient_checkpointing=False,
        )

        reloaded_model = PeftModel.from_pretrained(
            base_model,
            str(adapter_output),
            is_trainable=False,
        )

        reloaded_model.eval()

        if not hasattr(reloaded_model, "peft_config"):
            raise RuntimeError("Reloaded model has no PEFT configuration.")

        report["checks"]["adapter_reload"] = True
        report["status"] = "passed"

    except Exception as error:
        report["failed_stage"] = stage
        report["errors"].append(
            f"{type(error).__name__}: {error!r}"
        )
        report["traceback"] = traceback.format_exc()

    finally:
        reloaded_model = None
        base_model = None
        trainer = None
        model = None
        tokenizer = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if temporary_directory is not None:
            temporary_directory.cleanup()

        report["duration_seconds"] = round(
            time.time() - started, 2
        )

        if report_path:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return report
