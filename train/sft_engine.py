from sft_config import TrainConfig, TrainSettings
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig


class TrainEngine:
    # NOTE: Specify the HF type for the model and the tokenizer
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> None:
        self.tokenizer = tokenizer
        self.model = model

    @classmethod
    def _from_settings(cls, settings: TrainSettings) -> "TrainSettings":
        "Configure the model from the basemodel and causal_LM_settings settings"

        # Get the tokenizer and causal model
        tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            settings.base_model,
            dtype = settings.dtype,
            device_map = settings.device_map,
            attn_implementation = settings.attn_implementation
        )

        print(f"Device: {model.device}")
        print(f"DType: {model.dtype}")
    
        return cls(tokenizer, model)

    def train_model(self, settings: TrainSettings, train_data: Dataset, val_data: Dataset) -> dict:
        # return the data


        torch_dtype = self.model.dtype

        # NOTE: CHANGE THE FOLLOWING TO LOAD FROM SETTINGS

        args = SFTConfig(
            output_dir=checkpoint_dir,              # directory to save and repository id
            max_length=512,                         # max sequence length for model and packing of the dataset
            packing=False,                          # Groups multiple samples in the dataset into a single sequence
            num_train_epochs=8,                     # number of training epochs
            per_device_train_batch_size=4,          # batch size per device during training
            gradient_checkpointing=False,           # Caching is incompatible with gradient checkpointing
            optim="adamw_torch_fused",              # use fused adamw optimizer
            logging_steps=1,                        # log every step
            #save_strategy="epoch",                  # save checkpoint every epoch
            eval_strategy="epoch",                  # evaluate checkpoint every epoch
            learning_rate=learning_rate,            # learning rate
            fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
            bf16=True if torch_dtype == torch.bfloat16 else False,  # use bfloat16 precision
            lr_scheduler_type="constant",            # use constant learning rate scheduler
            push_to_hub=True,                        # push model to hub
            report_to="tensorboard",                 # report metrics to tensorboard
        )
