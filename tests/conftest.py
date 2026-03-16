import types
import pytest
from datasets import Dataset, DatasetDict
from unittest.mock import MagicMock, patch



@pytest.fixture
def args():
    return types.SimpleNamespace(
        model_name_or_path="gemma-3-270m-it",
        dataset_name="soynade-research/fineweb_synthetic",
        output_dir="./output/oolel-small",
        hub_model_id="",
        max_length=4096,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        report_to="none",
    )


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "</s>"
    return tok


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.fixture
def trainer(args, mock_tokenizer, mock_model):
    with patch("src.train.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("src.train.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
        from src.train import OolelTrainer
        return OolelTrainer(args)


@pytest.fixture
def valid_dataset():
    return DatasetDict({"train": Dataset.from_dict({"messages": ["hi"] * 5})})


@pytest.fixture
def invalid_dataset():
    return DatasetDict({"train": Dataset.from_dict({"text": ["hi"] * 5})})

