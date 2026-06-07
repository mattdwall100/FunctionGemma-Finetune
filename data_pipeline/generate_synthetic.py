import json
from typing import List


# read tool schemas
with open("data/schemas.jsonl", "r", encoding="utf-8") as f:
    TOOLS = [json.loads(line.strip()) for line in f if line.strip()]

arg_injections = {
    "internal_id": ["1", "2", "3", "4", "5", "one", "two", "three", "four", "five"],
    # Add more argument injections as needed for other tools
}

# system prompt
DEFAULT_SYSTEM_MSG = "You are a model named Alfred that can do function calling with the following functions"


def create_conversation(
    user_content: str, tool_name: str, tool_arguments: dict
) -> dict[str, List[dict]]:
    if not DEFAULT_SYSTEM_MSG:
        raise ValueError("DEFAULT_SYSTEM_MSG is not set. Please set it before creating conversations.")
    if not TOOLS:
        raise ValueError("TOOLS list is empty. Please load tool schemas before creating conversations.")

    return {
        "messages": [
            {"role": "developer", "content": DEFAULT_SYSTEM_MSG},
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_arguments,
                        },
                    }
                ],
            },
        ],
        "tools": TOOLS,
    }


import random

argument_variant_count = 3  # number of variations per argument

for tool in TOOLS:
    args = tool["function"]["parameters"]["properties"]  # dict form
    name = tool["function"]["name"]

    with open(f"data/processed/{name}.jsonl", "w", encoding="utf-8") as processed_f:
        with open(f"data/raw/{name}.txt", "r", encoding="utf-8") as raw_f:
            for line in raw_f:
                user_content = line.strip()
                if not user_content:
                    continue  # skip empties

                variants = [user_content]  # default to original user content if no args
                if args:
                    # If the tool requires arguments, we randomly inject choices from arg_injections
                    variants = [user_content] * argument_variant_count
                    for i, variant in enumerate(variants):
                        variant_args = {}
                        for arg_name in args.keys():
                            if arg_name in arg_injections.keys():
                                variant_args[arg_name] = random.choice(
                                    arg_injections[arg_name]
                                )
                            else:
                                raise ValueError(
                                    f"Warning: No injections found for argument '{arg_name}' in tool '{name}'. Using placeholder value."
                                )
                            # Inject argument choices into the user content
                            variants[i] = variant.format(**variant_args)

                conversations = [
                    create_conversation(variant, name, args) for variant in variants
                ]

                for conversation in conversations:
                    processed_f.write(json.dumps(conversation) + "\n")


from datasets import load_dataset

# Get the negative data from the cosmosai471/General_Conversation_Mixed_Dataset dataset, which is a mix of human and model 
# conversations that are not grounded in tool use. We will label these with empty tool calls to teach the model when NOT to call tools.

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("cosmosai471/General_Conversation_Mixed_Dataset", split="train")
ds = ds.select_columns(["text"])
ds = ds.map(lambda x: {"text": x["text"].strip().replace("Luna", "Alfred")})
ds = ds.map(lambda x: {"text": x["text"].replace("\u2019", "'")})

with open("data/processed/NO_TOOL.jsonl", "w", encoding="utf-8") as f:
    for item in ds:
        # json.dumps converts the dictionary to a string
        f.write(json.dumps(create_conversation(user_content=item["text"], tool_name="", tool_arguments={}), ensure_ascii=False) + "\n")
