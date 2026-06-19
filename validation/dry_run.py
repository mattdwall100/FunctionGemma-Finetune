from train.sft_config import TrainSettings, TrainConfig
from train.sft_engine import TrainEngine
from datasets import load_dataset

TRAIN_DATA_PATH = "data/final/final_train.jsonl"
SPLIT_RATIO = 0.85
SEED = 100

def run():
    config_manager = TrainConfig._from_yaml(config_file="configs/train.yaml")
    base_settings = config_manager.base_settings

    engine = TrainEngine._from_settings(settings=base_settings)

    dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
    data_splits = dataset.train_test_split(train_size=SPLIT_RATIO, seed=SEED)

    results = engine.train_model(base_settings, data_splits["train"], data_splits["test"])


if __name__ == "__main__":
    run()

