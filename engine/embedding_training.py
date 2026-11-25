def resolve_embedding_modules(model):
    """
    Dynamically resolve input embedding and output LM head module names.
    No architecture-specific module names are hardcoded.
    """
    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()

    if input_module is None:
        raise RuntimeError(
            "Cannot resolve input embedding module."
        )

    if output_module is None:
        raise RuntimeError(
            "Cannot resolve output embedding / LM head module."
        )

    try:
        named_modules = list(
            model.named_modules(remove_duplicate=False)
        )
    except TypeError:
        named_modules = list(model.named_modules())

    def find_names(target):
        return [
            name
            for name, module in named_modules
            if name and module is target
        ]

    input_names = find_names(input_module)
    output_names = find_names(output_module)

    if not input_names:
        raise RuntimeError(
            "Input embedding module was not found in model.named_modules()."
        )

    if not output_names:
        raise RuntimeError(
            "Output embedding / LM head module was not found "
            "in model.named_modules()."
        )

    # If both APIs return the exact same module object, only one
    # modules_to_save entry is necessary.
    if input_module is output_module:
        module_names = [input_names[0]]
    else:
        module_names = sorted(set(input_names + output_names))

    shared_weight = (
        hasattr(input_module, "weight")
        and hasattr(output_module, "weight")
        and input_module.weight is output_module.weight
    )

    config_tied = bool(
        getattr(model.config, "tie_word_embeddings", False)
    )

    return {
        "modules_to_save": module_names,
        "tied_embeddings": shared_weight or config_tied,
        "input_modules": input_names,
        "output_modules": output_names,
    }


def configure_embedding_training(model, lora_config):
    """
    Configure PEFT to train and save the resolved embedding and LM head.
    """
    resolved = resolve_embedding_modules(model)

    lora_config.modules_to_save = resolved["modules_to_save"]

    if resolved["tied_embeddings"]:
        lora_config.ensure_weight_tying = True

    return resolved
