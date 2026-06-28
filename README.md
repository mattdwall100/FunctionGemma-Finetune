Finetuning Function Gemma to act as the sole tool-routing mechanism in local-AI-assistant pipeline.

SET-UP:

git clone ...
cd FunctionGemma-Finetune
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


GENERATE DATA:

1. store tools in tools.py
2. Generate schemas:
   > > python .\data_pipeline\schemas\generate_schemas.py
3. Store synthetic data in .\data\raw\tool_name.txt line by line.
   If any argument is needed, replace in-line with {argument_name}.
   add a line to [CONFIG LOCATION] to specify per-tool argument variants that will be injected during data processing.
4. Create processed data with argument injection variants:
   > > python data_pipeline.generate_synthetic
5. Get final split:
   > > python -m data_pipeline.final_split

# $env:PYTHONUTF8 = "1"
hf auth login 
--extra-index-url https://download.pytorch.org/whl/cu128

# Upload final_merged_export to HF 
export HF_TOKEN=...
python -m scripts.upload_merged_model --repo-id your-username/your-model-name

# Convert final_merged_export to GGUF and upload to HF
export HF_TOKEN=...
python -m scripts.upload_gguf_model \
  --repo-id your-username/your-model-name-GGUF \
  --converter /path/to/llama.cpp/convert_hf_to_gguf.py

# ... GGUF without convertion
python -m scripts.upload_gguf_model \
  --repo-id your-username/your-model-name-GGUF \
  --outfile gguf/functiongemma-merged.f16.gguf \
  --skip-convert

# Download llama.cpp repo to use convert_hf_to_gguf.py
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
python3 -m pip install -r requirements/requirements-convert_hf_to_gguf.txt

# Use via python in CLI
python3 convert_hf_to_gguf.py /path/to/huggingface/model --outfile model.gguf