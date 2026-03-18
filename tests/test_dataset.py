import pytest
from datasets import Dataset, DatasetDict
from unittest.mock import patch


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


def test_load_data_succeeds_with_valid_dataset(trainer, valid_dataset):
    with patch("src.train.load_dataset", return_value=valid_dataset):
        result = trainer.load_data()
        assert "train" in result
        assert "messages" in result["train"].column_names


def test_missing_column_checked_across_all_splits(trainer):
    """load_data() loops over ALL splits — bad validation split must also raise."""
    bad = DatasetDict(
        {
            "train": Dataset.from_dict({"messages": ["hi"] * 3}),
            "validation": Dataset.from_dict({"text": ["hi"] * 3}),
        }
    )
    with patch("src.train.load_dataset", return_value=bad):
        with pytest.raises(ValueError, match="validation"):
            trainer.load_data()
