import re
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)


BASE_MODEL = "google/functiongemma-270m-it"
CHECKPOINT = "checkpoints/checkpoint-282"
TEST_DATA_PATH = "data/final/final_test.jsonl"

START_FUNCTION_CALL = "<start_function_call>"
END_FUNCTION_CALL = "<end_function_call>"
END_OF_TURN = "<end_of_turn>"

CALL_RE = re.compile(r"<start_function_call>\s*call:([a-zA-Z_][a-zA-Z0-9_]*)")


class StopOnAnySubsequence(StoppingCriteria):
    def __init__(self, stop_sequences):
        self.stop_sequences = stop_sequences

    def __call__(self, input_ids, scores, **kwargs):
        ids = input_ids[0].tolist()

        for stop_ids in self.stop_sequences:
            if len(ids) >= len(stop_ids) and ids[-len(stop_ids) :] == stop_ids:
                return True

        return False


def get_expected_tool(example):
    assistant_msg = example["messages"][-1]

    if "tool_calls" not in assistant_msg:
        return None

    return assistant_msg["tool_calls"][0]["function"]["name"]


def get_predicted_tool(output: str):
    """
    Ollama-compatible parser logic:

    - If <end_of_turn> appears before any <start_function_call>, predict no tool.
    - Otherwise, parse the first function call.
    - If there is no function call, predict no tool.
    """
    end_pos = output.find(END_OF_TURN)
    call_pos = output.find(START_FUNCTION_CALL)

    if call_pos == -1:
        return None

    if end_pos != -1 and end_pos < call_pos:
        return None

    match = CALL_RE.search(output, pos=call_pos)

    if match is None:
        return None

    return match.group(1)


def evaluate_tool_accuracy(model_path: str, name: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    stop_sequences = [
        tokenizer.encode(END_OF_TURN, add_special_tokens=False),
        tokenizer.encode(END_FUNCTION_CALL, add_special_tokens=False),
    ]

    stopping_criteria = StoppingCriteriaList([StopOnAnySubsequence(stop_sequences)])

    test_data = load_dataset("json", data_files=TEST_DATA_PATH, split="train")

    correct = 0
    total = 0

    for idx, item in enumerate(test_data):
        prompt_messages = item["messages"][:-1]
        expected_tool = get_expected_tool(item)

        inputs = tokenizer.apply_chat_template(
            prompt_messages,
            tools=item["tools"],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                max_new_tokens=128,
                do_sample=False,
            )

        output = tokenizer.decode(
            out[0][len(inputs["input_ids"][0]) :],
            skip_special_tokens=False,
        )

        predicted_tool = get_predicted_tool(output)
        is_correct = predicted_tool == expected_tool

        correct += int(is_correct)
        total += 1

        print(("=" * 40) + ("CORRECT" if is_correct else "FALSE") + ("=" * 40))
        print(f"Example {idx + 1}")

        print("User:")
        print(item["messages"][-2]["content"])

        print("\nExpected tool:")
        print(expected_tool)

        print("\nPredicted tool:")
        print(predicted_tool)

        print(
            "\nTruncated output: (before stop condition (end of turn or end function calll tokens ))"
        )
        print(output)

    print(f"\n{name}")
    print(f"Correct: {correct} / {total}")
    print(f"Tool accuracy: {correct / total:.2%}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # evaluate_tool_accuracy(BASE_MODEL, "Base model")
    evaluate_tool_accuracy(CHECKPOINT, "Checkpoint 282")
