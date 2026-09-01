import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.training_config import build_sft_config_kwargs, validate_dataset_schema


class OolelTrainer:
    def __init__(self, args):
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            torch_dtype="auto",
            attn_implementation=args.attn_implementation,
        )

    def load_data(self):
        dataset = load_dataset(self.args.dataset_name)
        return validate_dataset_schema(dataset)

    def train(self):
        dataset = self.load_data()

        training_args = SFTConfig(
            **build_sft_config_kwargs(self.args, dataset, self.model.dtype)
        )

        # SFTTrainer handles chat-template formatting automatically when
        # the dataset contains a 'messages' column (list of dicts with role/content).
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            processing_class=self.tokenizer,
        )

        trainer.train()
        trainer.save_model(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)

        if self.args.hub_model_id:
            trainer.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oolel-small training pipeline")

    # Model & data
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./oolel-small-finetuned")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")

    # Sequence
    parser.add_argument("--max_length", type=int, default=4096)

    # Batch / accumulation
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", type=bool, default=False)

    # Optimisation
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw_torch_fused")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--report_to", type=str, default="tensorboard")

    args = parser.parse_args()
    OolelTrainer(args).train()
