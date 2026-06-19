from .sft_config import TrainSettings
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer


class TrainEngine:
    # NOTE: Specify the HF type for the model and the tokenizer
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> None:
        self.tokenizer = tokenizer
        self.model = model

    @classmethod
    def _from_settings(cls, settings: TrainSettings) -> "TrainEngine":
        "Configure the model from the base_model and causal_LM_settings settings"

        # Get the tokenizer and causal model
        tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            settings.base_model,
            dtype=settings.dtype,
            device_map=settings.device_map,
            attn_implementation=settings.attn_implementation,
        )

        print(f"Device: {model.device}")
        print(f"DType: {model.dtype}")

        return cls(tokenizer, model)

    # NOTE: Find way to use TrainOutput from transformers as return type
    def train_model(
        self, settings: TrainSettings, train_data: Dataset, val_data: Dataset
    ) -> dict:
        # Create the SFT Config and Trainer
        torch_dtype = self.model.dtype
        args = SFTConfig(
            use_cpu=settings.use_cpu, # Need if gpu doesnt support datatype
            output_dir=settings.output_dir,  # directory to save and repository id
            max_length=settings.max_length,  # max sequence length for model and packing of the dataset
            packing=settings.packing,  # Groups multiple samples in the dataset into a single sequence
            num_train_epochs=settings.num_train_epochs,  # number of training epochs
            per_device_train_batch_size=settings.per_device_train_batch_size,  # batch size per device during training
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            gradient_checkpointing=settings.gradient_checkpointing,  # Caching is incompatible with gradient checkpointing
            optim=settings.optim,  # use fused adamw optimizer
            logging_steps=settings.logging_steps,  # log every step
            save_strategy=settings.save_strategy,  # save checkpoint every epoch
            eval_strategy=settings.eval_strategy,  # evaluate checkpoint every epoch
            per_device_eval_batch_size=settings.per_device_eval_batch_size,
            eval_on_start=settings.eval_on_start,
            learning_rate=settings.learning_rate,  # learning rate
            fp16= True if torch_dtype == torch.float16 else False,  # use float16 precision
            bf16= True if torch_dtype == torch.bfloat16 else False,  # use bfloat16 precision
            lr_scheduler_type=settings.lr_scheduler_type,  # use constant learning rate scheduler
            push_to_hub=settings.push_to_hub,  # push model to hub
            report_to=settings.report_to,  # report metrics to tensorboard
        )
        # Create Trainer object
        trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=train_data,
            eval_dataset=val_data,
            processing_class=self.tokenizer,
        )

        # Start training, the model will be automatically saved to the Hub and the output directory
        train_results = trainer.train()

        # Save the final model again to the Hugging Face Hub
        trainer.save_model()

        return train_results


