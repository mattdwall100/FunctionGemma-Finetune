from pydantic import BaseModel
from pathlib import Path
import yaml

CONFIG_PATH = "configs/train.yaml"
CHECKPOINT_DIR = "checkpoints"

# NOTE: specify all fields and defaults here


class TrainSettings(BaseModel):
    # HF model address
    base_model: str = "google/functiongemma-270m-it"

    # Casual LM settings
    dtype: str = "auto"
    device_map: str = "auto"
    attn_implementation: str = "eager"

    # SFT hyperparameters (defaults are from Google FunctionGemma FT guide)
    use_cpu: bool = False # needed if auto data type not compatible with cpu
    output_dir: str = ("checkpoints",)  # directory to save and repository id
    max_length: int = (512,)  # max sequence length for model and packing of the dataset
    packing: bool = (
        False,
    )  # Groups multiple samples in the dataset into a single sequence
    num_train_epochs: int = (8,)  # number of training epochs
    per_device_train_batch_size: int = (2,)  # batch size per device during training
    gradient_accumulation_steps: int = (2,)
    gradient_checkpointing: bool = (
        False,
    )  # Caching is incompatible with gradient checkpointing
    optim: str = ("adamw_torch_fused",)  # use fused adamw optimizer
    logging_steps: int = (1,)  # log every step
    save_strategy: str = ("epoch",)  # save checkpoint every epoch
    eval_strategy: str = ("epoch",)  # evaluate checkpoint every epoch
    per_device_eval_batch_size: int = (2,)
    eval_on_start: bool = False
    learning_rate: float = (5e-5,)  # learning rate
    lr_scheduler_type: str = ("constant",)  # use constant learning rate scheduler
    push_to_hub: bool = (False,)  # push model to hub
    report_to: str = ("tensorboard",)  # report metrics to tensorboard


class TrainConfig:
    def __init__(self, settings: TrainSettings):
        self.__base_settings = settings

    @classmethod
    def _from_yaml(cls, config_file: str = CONFIG_PATH) -> "TrainConfig":
        """Constructor method loads training settings from yaml"""

        path = Path(CONFIG_PATH)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        config = raw.get("settings", {})

        settings_kwargs = {}
        settings_kwargs["base_model"] = config.get("base_model", "")
        settings_kwargs.update(config.get("causal_lm_settings", {}))
        settings_kwargs.update(config.get("sft_settings", {}))
        settings_kwargs.update(config.get("data", {}))

        try:
            settings = TrainSettings(**settings_kwargs)
        except Exception as e:
            raise ValueError(f"Invalid config for training: {e}") from e

        return cls(settings)

    @property
    def base_settings(self) -> TrainSettings:
        return self.__base_settings

    def get_pathched(self, patch_kwargs: dict) -> TrainSettings:
        settings_kwargs = self.base_settings.model_dump()
        settings_kwargs.update(patch_kwargs)

        try:
            patched_settings = TrainSettings(**settings_kwargs)
        except Exception as e:
            raise ValueError(f"Invalid config for training: {e}") from e

        return patched_settings
