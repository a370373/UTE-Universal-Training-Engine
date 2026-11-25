import gc
import json
import math
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer
from engine.model_loader import load_base_model
from engine.embedding_training import configure_embedding_training
from peft import LoraConfig, get_peft_model


def run_smoke_test(
    model_path,
    lora_config,
    dtype=torch.float32,
    max_length=128,
    cpu_threads=4,
    report_path=None,
    quantization=None,
    train_embeddings=False,
):
    started = time.time()

    report = {
        "test": "training_smoke_test",
        "model_path": str(model_path),
        "quantization": (
            quantization["metadata"]
            if quantization is not None
            else {"method": "LoRA", "quantization": "none"}
        ),
        "status": "failed",
        "checks": {},
        "errors": [],
    }

    stage = "initialization"
    model = None
    peft_model = None
    optimizer = None

    try:
        torch.set_num_threads(cpu_threads)

        stage = "tokenizer_loading"
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False,
            local_files_only=Path(model_path).exists(),
        )

        stage = "chat_template_check"
        if not tokenizer.chat_template:
            raise RuntimeError("Tokenizer has no chat template.")

        messages = [
            {
                "role": "user",
                "content": "請用一句話介紹人工智慧。"
            },
            {
                "role": "assistant",
                "content": "人工智慧是讓電腦執行需要人類智慧的任務的技術。"
            },
        ]

        stage = "tokenization"
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_dict=True,
        )

        input_ids = encoded["input_ids"]

        if input_ids.numel() == 0:
            raise RuntimeError("Tokenization produced an empty sequence.")

        report["checks"]["tokenization"] = True
        report["input_tokens"] = int(input_ids.shape[-1])

        stage = "model_loading"
        model = load_base_model(
            model_path=model_path,
            dtype=dtype,
            quantization=quantization,
            gradient_checkpointing=False,
        )

        if train_embeddings:
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

        stage = "lora_injection"
        peft_model = get_peft_model(model, lora_config)
        peft_model.train()

        input_device = model.get_input_embeddings().weight.device
        input_ids = input_ids.to(input_device)

        report["checks"]["lora_injection"] = True

        total_params = 0
        trainable_params = 0

        for parameter in peft_model.parameters():
            total_params += parameter.numel()
            if parameter.requires_grad:
                trainable_params += parameter.numel()

        if trainable_params <= 0:
            raise RuntimeError("No trainable parameters found.")

        if trainable_params >= total_params:
            raise RuntimeError("Unexpected trainable parameter count.")

        report["total_parameters"] = total_params
        report["trainable_parameters"] = trainable_params
        report["trainable_ratio"] = trainable_params / total_params

        labels = input_ids.clone()

        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad],
            lr=1e-4,
        )

        optimizer.zero_grad(set_to_none=True)

        stage = "forward_pass"
        output = peft_model(
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
        )

        loss = output.loss

        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Forward pass produced invalid loss.")

        report["loss"] = float(loss.detach().item())
        report["checks"]["forward"] = True

        stage = "backward_pass"
        loss.backward()

        gradient_count = 0

        for parameter in peft_model.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("Non-finite gradient detected.")
                gradient_count += 1

        if gradient_count == 0:
            raise RuntimeError("No gradients were produced.")

        report["gradient_tensors"] = gradient_count
        report["checks"]["backward"] = True

        stage = "optimizer_step"
        optimizer.step()

        report["checks"]["optimizer_step"] = True
        report["status"] = "passed"

    except Exception as error:
        report["failed_stage"] = stage
        report["errors"].append(
            f"{type(error).__name__}: {error!r}"
        )
        report["traceback"] = traceback.format_exc()

    finally:
        if optimizer is not None:
            del optimizer
        if peft_model is not None:
            del peft_model
        if model is not None:
            del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report["duration_seconds"] = round(time.time() - started, 2)

    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return report
