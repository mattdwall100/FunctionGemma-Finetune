import os
import random
from datasets import load_dataset, concatenate_datasets

# NOTE: STORE IN CONFIG LATER

# Path configurations
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "data/final"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "final_train.jsonl")
TEST_FILE = os.path.join(OUTPUT_DIR, "final_test.jsonl")

# Configuration configurations
SPLIT_RATIO = 0.8  # Train/Test Split
NEGATIVE_DATA_RATIO = 0.25  # ⚠️ 25% of the TOTAL final dataset will be NO_TOOL
SEED = 42
random.seed(SEED)


def compile_balanced_dataset():
    jsonl_files = [
        file_name
        for file_name in os.listdir(PROCESSED_DIR)
        if file_name.endswith(".jsonl")
    ]

    print("Loading tool files...")
    # get paths
    tool_paths = [
        os.path.join(PROCESSED_DIR, file_name)
        for file_name in jsonl_files
        if file_name != "NO_TOOL.jsonl"
    ]
    no_tool_path = [os.path.join(PROCESSED_DIR, "NO_TOOL.jsonl")]

    # load
    tool_ds = load_dataset("json", data_files=tool_paths, split="train")
    no_tool_ds = load_dataset("json", data_files=no_tool_path, split="train")

    print("Balancing negative data ratio...")

    tool_ds_size = len(tool_ds)
    print(f"Total active tool-calling examples aggregated: {tool_ds_size}")

    # R = neg / (neg + pos)
    # neg = ( R / (1 - R) ) * pos
    no_tool_ds_size = int(
        tool_ds_size * (NEGATIVE_DATA_RATIO / (1 - NEGATIVE_DATA_RATIO))
    )

    if len(no_tool_ds) < no_tool_ds_size:
        print(
            f"Warning: Not enough negative samples to achieve ratio {NEGATIVE_DATA_RATIO}"
        )
        no_tool_ds_size = len(no_tool_ds)
        print(f"Achieved ratio: {no_tool_ds_size / tool_ds_size}")

    # Downsample to meet ratio if possible (no shuffle because we favour the earlier examples)
    downsampled_no_tool_ds = no_tool_ds.select(range(no_tool_ds_size))

    print("Combining and shuffling datasets...")
    target_feature_schemas = tool_ds.features
    aligned_no_tool_ds = downsampled_no_tool_ds.cast(target_feature_schemas)

    master_ds = concatenate_datasets([tool_ds, aligned_no_tool_ds])
    master_ds = master_ds.shuffle(seed=SEED)

    print(f"Creating Train/Test split ({SPLIT_RATIO})...")

    ds_splits = master_ds.train_test_split(train_size=SPLIT_RATIO, seed=SEED)

    print("Exporting the clean train/test datasets...")

    ds_splits["train"].to_json(os.path.join(OUTPUT_DIR, "final_train.jsonl"))
    ds_splits["test"].to_json(os.path.join(OUTPUT_DIR, "final_test.jsonl"))

    print("Finished.")
    print(f"Train size: {len(ds_splits['train'])}")
    print(f"Test size: {len(ds_splits['test'])}")


if __name__ == "__main__":
    compile_balanced_dataset()
