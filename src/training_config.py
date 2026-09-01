from typing import Any


def validate_dataset_schema(dataset):
    """Validate the dataset contract required by the chat trainer."""
    if "train" not in dataset:
        raise ValueError("Dataset is missing the required 'train' split.")
    if len(dataset["train"]) == 0:
        raise ValueError("The 'train' split must contain at least one example.")

    for split_name, split in dataset.items():
        if "messages" not in split.column_names:
            raise ValueError(
                f"Split '{split_name}' is missing the 'messages' column. "
                "Dataset must be pre-formatted as a list of chat messages."
            )

    return dataset


def build_sft_config_kwargs(args, dataset, torch_dtype: Any) -> dict[str, Any]:
    """Build deterministic TRL configuration without starting model training."""
    has_validation = "validation" in dataset
    dtype_name = str(torch_dtype).removeprefix("torch.")

    return {
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "optim": args.optim,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "fp16": dtype_name == "float16",
        "bf16": dtype_name == "bfloat16",
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "eval_strategy": "epoch" if has_validation else "no",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": has_validation,
        "report_to": args.report_to,
        "hub_model_id": args.hub_model_id,
        "push_to_hub": bool(args.hub_model_id),
        "dataset_kwargs": {
            "add_special_tokens": False,
            "append_concat_token": True,
        },
    }
