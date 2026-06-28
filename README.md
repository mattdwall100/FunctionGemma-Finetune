Finetuning Function Gemma to act as the sole tool-routing mechanism in local-AI-assistant pipeline.

SET-UP:

git clone ...
cd FunctionGemma-Finetune
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


USE:

1. store tools in tools.py
2. Generate schemas:
   > > python .\data_pipeline\schemas\generate_schemas.py
3. Store synthetic data:
   in .\data\raw\tool_name.txt
   line by line.
   If any argument is needed, replace in-line with {argument_name}.
   add a line to [CONFIG LOCATION] to specify per-tool argument variants that will be injected during data processing.
4. Create processed data with argument injection variants:
   > > python .\data_pipeline\generate_synthetic.py

# $env:PYTHONUTF8 = "1"
hf auth login 
--extra-index-url https://download.pytorch.org/whl/cu128