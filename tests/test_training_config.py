from types import SimpleNamespace

from src.training_config import build_sft_config_kwargs


def make_args(**overrides):
    values = {
        "output_dir": "./output/oolel-small",
        "max_length": 4096,
        "epochs": 3,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": False,
        "optim": "adamw_torch_fused",
        "learning_rate": 2e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "report_to": "none",
        "hub_model_id": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_config_without_validation_stays_local_and_uses_fp16():
    config = build_sft_config_kwargs(make_args(), {"train": object()}, "torch.float16")

    assert config["eval_strategy"] == "no"
    assert config["load_best_model_at_end"] is False
    assert config["push_to_hub"] is False
    assert (config["fp16"], config["bf16"]) == (True, False)
    assert config["max_length"] == 4096
    assert config["dataset_kwargs"] == {
        "add_special_tokens": False,
        "append_concat_token": True,
    }


def test_validation_and_hub_configuration_enable_related_behaviour():
    config = build_sft_config_kwargs(
        make_args(hub_model_id="soynade-research/oolel-small"),
        {"train": object(), "validation": object()},
        "torch.bfloat16",
    )

    assert config["eval_strategy"] == "epoch"
    assert config["load_best_model_at_end"] is True
    assert config["push_to_hub"] is True
    assert config["hub_model_id"] == "soynade-research/oolel-small"
    assert (config["fp16"], config["bf16"]) == (False, True)
