from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


def inspect_compatibility(model_path):
    model_path = str(model_path)
    local = Path(model_path).exists()

    report = {
        "model_path": model_path,
        "checks": {},
        "warnings": [],
        "errors": [],
    }

    try:
        config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=local,
            trust_remote_code=False,
        )
        report["model_type"] = getattr(config, "model_type", None)
        report["architectures"] = getattr(config, "architectures", None)
        report["config_vocab_size"] = getattr(config, "vocab_size", None)
        report["checks"]["config_loaded"] = True
    except Exception as error:
        report["errors"].append(f"Config loading failed: {error}")
        return report

    architectures = report["architectures"] or []

    if not report["model_type"]:
        report["errors"].append("Missing model_type.")

    if architectures and not any(
        "CausalLM" in architecture for architecture in architectures
    ):
        report["errors"].append(
            f"Architecture does not appear to be CausalLM: {architectures}"
        )

    report["checks"]["causal_lm_architecture"] = (
        not report["errors"]
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local,
            trust_remote_code=False,
        )

        tokenizer_size = len(tokenizer)
        config_size = report["config_vocab_size"]

        report["tokenizer_class"] = tokenizer.__class__.__name__
        report["tokenizer_vocab_size"] = tokenizer_size
        report["pad_token_id"] = tokenizer.pad_token_id
        report["eos_token_id"] = tokenizer.eos_token_id
        report["has_chat_template"] = bool(tokenizer.chat_template)

        report["checks"]["tokenizer_loaded"] = True

        if tokenizer.eos_token_id is None:
            report["errors"].append("Tokenizer has no EOS token.")

        if not tokenizer.chat_template:
            report["errors"].append("Tokenizer has no chat template.")

        if config_size is not None:
            if tokenizer_size > config_size:
                report["errors"].append(
                    "Tokenizer vocabulary exceeds config vocabulary."
                )
            elif tokenizer_size != config_size:
                report["warnings"].append(
                    "Tokenizer and config vocabulary sizes differ. "
                    "Embedding dimensions must be checked after model loading."
                )

        report["checks"]["tokenizer_basic"] = (
            tokenizer.eos_token_id is not None
            and bool(tokenizer.chat_template)
            and (
                config_size is None
                or tokenizer_size <= config_size
            )
        )

    except Exception as error:
        report["checks"]["tokenizer_loaded"] = False
        report["errors"].append(f"Tokenizer loading failed: {error}")

    report["compatible"] = len(report["errors"]) == 0

    return report
