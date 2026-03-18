import pytest
from unittest.mock import patch


def test_pad_token_falls_back_to_eos(trainer):
    assert trainer.tokenizer.pad_token == "</s>"


def test_model_is_attached(trainer, mock_model):
    assert trainer.model is mock_model


def test_tokenizer_is_attached(trainer, mock_tokenizer):
    assert trainer.tokenizer is mock_tokenizer


def test_init_raises_when_tokenizer_fails(args):
    with patch(
        "src.train.AutoTokenizer.from_pretrained", side_effect=OSError("not found")
    ):
        from src.train import OolelTrainer

        with pytest.raises(OSError, match="not found"):
            OolelTrainer(args)


def test_init_raises_when_model_fails(args, mock_tokenizer):
    with (
        patch("src.train.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch(
            "src.train.AutoModelForCausalLM.from_pretrained",
            side_effect=OSError("model not found"),
        ),
    ):
        from src.train import OolelTrainer

        with pytest.raises(OSError, match="model not found"):
            OolelTrainer(args)
