print("hello1")

from transformers import AutoModelForCausalLM, AutoTokenizer
print("hello2")
from peft import PeftModel
print("hello3")


OUTPUT_DIR = "./final_merged_export"

# Load and merge weights
base = AutoModelForCausalLM.from_pretrained("google/functiongemma-270m-it")
model = PeftModel.from_pretrained(base, "checkpoints/checkpoint-282")
merged_model = model.merge_and_unload()

# Save everything together to seperate dir
merged_model.save_pretrained(OUTPUT_DIR)

tokenizer = AutoTokenizer.from_pretrained("google/functiongemma-270m-it")
tokenizer.save_pretrained(OUTPUT_DIR)  # This prevents the tokenizer from changing