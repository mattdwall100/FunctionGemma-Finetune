import json

from typing import List
import random
from datasets import load_dataset


def get_data():

    # read tool schemas
    with open("data/schemas.jsonl", "r", encoding="utf-8") as f:
        TOOLS = [json.loads(line.strip()) for line in f if line.strip()]

    arg_injections = {
        "internal_id": [
            ("1", "1"),
            ("2", "2"),
            ("3", "3"),
            ("4", "4"),
            ("5", "5"),
            ("one", "1"),
            ("two", "2"),
            ("three", "3"),
            ("four", "4"),
            ("five", "5"),
        ],
        # Add more argument injections as needed for other tools
    }

    # system prompt
    DEFAULT_SYSTEM_MSG = "You are a model named Alfred that can do function calling with the following functions"

    def create_conversation(
        user_content: str, tool_name: str, tool_arguments: dict, tool_called=True
    ) -> dict[str, List[dict]]:
        if not DEFAULT_SYSTEM_MSG:
            raise ValueError(
                "DEFAULT_SYSTEM_MSG is not set. Please set it before creating conversations."
            )
        if not TOOLS:
            raise ValueError(
                "TOOLS list is empty. Please load tool schemas before creating conversations."
            )

        messages = [
            {"role": "developer", "content": DEFAULT_SYSTEM_MSG},
            {"role": "user", "content": user_content},
        ]

        if tool_name:
            messages.append(
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
            )
        else:
            messages.append({"role": "assistant", "content": ""})
        return {
            "messages": messages,
            "tools": TOOLS,
        }

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

                    variants = [
                        (user_content, args)
                    ]  # default to original user content if no args
                    if args:
                        # If the tool requires arguments, we randomly inject choices from arg_injections
                        variants = [user_content] * argument_variant_count
                        for i, variant in enumerate(variants):
                            variant_args = {}
                            mapped_variant_args = {}
                            for arg_name in args.keys():
                                if arg_name in arg_injections.keys():
                                    arg_map_choice = random.choice(
                                        arg_injections[arg_name]
                                    )
                                    variant_args[arg_name] = arg_map_choice[0]
                                    mapped_variant_args[arg_name] = arg_map_choice[1]
                                else:
                                    raise ValueError(
                                        f"Warning: No injections found for argument '{arg_name}' in tool '{name}'. Using placeholder value."
                                    )
                                # Inject argument choices into the user content
                                variants[i] = (
                                    variant.format(**variant_args),
                                    mapped_variant_args,
                                )

                    conversations = [
                        create_conversation(variant[0], name, variant[1])
                        for variant in variants
                    ]

                    for conversation in conversations:
                        processed_f.write(json.dumps(conversation) + "\n")

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
            f.write(
                json.dumps(
                    create_conversation(
                        user_content=item["text"], tool_name="", tool_arguments={}
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    get_data()
