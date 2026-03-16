from unittest.mock import MagicMock, patch
import pytest

def test_pad_token_falls_back_to_eos(trainer):
    assert trainer.tokenizer.pad_token == "</s>"


def test_model_is_attached(trainer, mock_model):
    assert trainer.model is mock_model

