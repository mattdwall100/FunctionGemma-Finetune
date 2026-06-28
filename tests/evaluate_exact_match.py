from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


BASE_MODEL = "google/functiongemma-270m-it"
CHECKPOINT = "checkpoints/checkpoint-2256"
TEST_DATA_PATH = "data/final/final_test.jsonl"


def build_prompt_inputs(tokenizer, example):
    """
    Build model input from all messages except the final assistant answer.
    """
    return tokenizer.apply_chat_template(
        example["messages"][:-1],
        tools=example["tools"],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def build_expected_ids(tokenizer, example, prompt_length):
    """
    Build the full expected conversation, then remove the prompt tokens.
    """
    full_inputs = tokenizer.apply_chat_template(
        example["messages"],
        tools=example["tools"],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    return full_inputs["input_ids"][0][prompt_length:]


def move_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def evaluate_exact_match(model_path: str, name: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    test_data = load_dataset("json", data_files=TEST_DATA_PATH, split="train")

    correct = 0
    total = 0

    for example in test_data:
        prompt_inputs = build_prompt_inputs(tokenizer, example)
        prompt_length = prompt_inputs["input_ids"].shape[-1]

        expected_ids = build_expected_ids(tokenizer, example, prompt_length)

        prompt_inputs = move_to_device(prompt_inputs, model.device)
        expected_ids = expected_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **prompt_inputs,
                max_new_tokens=expected_ids.shape[-1],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][prompt_length:]

        is_correct = torch.equal(
            generated_ids.cpu(),
            expected_ids.cpu(),
        )

        correct += int(is_correct)
        total += 1

        if not is_correct:
            print("=" * 80)
            print("USER MESSAGE:")
            print(example["messages"][-2]["content"])

            print("\nEXPECTED:")
            print(tokenizer.decode(expected_ids, skip_special_tokens=False))

            print("\nPREDICTED:")
            print(tokenizer.decode(generated_ids, skip_special_tokens=False))

    accuracy = correct / total if total else 0

    print(f"\n{name}")
    print(f"Correct: {correct}/{total}")
    print(f"Exact match accuracy: {accuracy:.2%}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # evaluate_exact_match(BASE_MODEL, "Base model")
    evaluate_exact_match(CHECKPOINT, "Checkpoint 2256")
