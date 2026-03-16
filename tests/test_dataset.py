from unittest.mock import MagicMock, patch
import pytest

def test_dataset_has_messages_column(valid_dataset):
    assert "messages" in valid_dataset["train"].column_names


def test_dataset_train_split_present(valid_dataset):
    assert "train" in valid_dataset


def test_dataset_not_empty(valid_dataset):
    assert len(valid_dataset["train"]) > 0


def test_missing_messages_column_raises(trainer, invalid_dataset):
    with patch("src.train.load_dataset", return_value=invalid_dataset):
        with pytest.raises(ValueError, match="missing the 'messages' column"):
            trainer.load_data()

