from transformers.utils import get_json_schema
from tools import *

tool_list = [
    get_papers,
    get_summary,
    stage_paper,
    print_paper,
    get_time,
    get_date,
    list_titles,
    get_staged_id,
]

TOOLS = [get_json_schema(tool) for tool in tool_list]

# SAVE TO JSONL FILE FOR TRAINING

import json

# Write tools to data/schemas.jsonl
with open("data/schemas.jsonl", "w", encoding="utf-8") as f:
    for item in TOOLS:
        # json.dumps converts the dictionary to a string
        f.write(json.dumps(item) + "\n")
