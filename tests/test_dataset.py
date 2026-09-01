import pytest

from src.training_config import validate_dataset_schema


class Split:
    def __init__(self, columns=("messages",), size=1):
        self.column_names = columns
        self.size = size

    def __len__(self):
        return self.size


def test_valid_dataset_schema_is_returned_unchanged():
    dataset = {"train": Split(), "validation": Split()}

    assert validate_dataset_schema(dataset) is dataset


@pytest.mark.parametrize(
    ("dataset", "message"),
    [
        ({"validation": Split()}, "required 'train' split"),
        ({"train": Split(size=0)}, "at least one example"),
        (
            {"train": Split(), "validation": Split(columns=("text",))},
            "Split 'validation'.*messages",
        ),
    ],
)
def test_invalid_dataset_contract_is_rejected(dataset, message):
    with pytest.raises(ValueError, match=message):
        validate_dataset_schema(dataset)
