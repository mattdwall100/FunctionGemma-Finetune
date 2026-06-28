from datasets import load_dataset
from transformers import AutoTokenizer


BASE_MODEL = "google/functiongemma-270m-it"
TEST_DATA_PATH = "data/final/final_train.jsonl"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
data = load_dataset("json", data_files=TEST_DATA_PATH, split="train")

lengths = []

for ex in data:
    encoded = tokenizer.apply_chat_template(
        ex["messages"],
        tools=ex["tools"],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
    )

    token_length = len(encoded["input_ids"])
    lengths.append(token_length)

print("min:", min(lengths))
print("max:", max(lengths))
print("avg:", sum(lengths) / len(lengths))
print("over 512:", sum(x > 512 for x in lengths), "/", len(lengths))
print("over 1024:", sum(x > 1024 for x in lengths), "/", len(lengths))
print("over 2048:", sum(x > 2048 for x in lengths), "/", len(lengths))
print("over 4096:", sum(x > 4096 for x in lengths), "/", len(lengths))
