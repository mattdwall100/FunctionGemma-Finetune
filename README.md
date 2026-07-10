# FunctionGemma-Finetune

Fine-tuning a **270M-parameter** model into the tool-routing brain of a local voice assistant.
The model reads a user utterance and the available tool schemas, and decides **which tool to
call — or, just as importantly, to call none and let the assistant just talk.**

Fine-tuning took base tool-call accuracy from **63.5% → 99.5%** on a 332-example held-out test
set, on a single machine in **~19 minutes**. The merged model is published to the Hugging Face
Hub and runs in production as the router in
[local-assistant-server](https://github.com/mattdwall100/local-assistant-server).

## Results

| | Base `google/functiongemma-270m-it` | Fine-tuned |
| --- | ---: | ---: |
| **Tool-call accuracy** (332 held-out examples) | **63.5%** | **99.5%** |

Reproduce with [`tests/evaluate.py`](tests/evaluate.py) (tool-name accuracy, greedy decoding,
Ollama-compatible output parser). Source of the headline figures: commit `c75c2f3`. Uncomment
the `evaluate_tool_accuracy(BASE_MODEL, ...)` line to score the base model alongside a checkpoint.

## Why

The assistant used to route tools with hand-written parsing. That is brittle: every new phrasing
is a new `if`. Replacing it with a small fine-tuned model means routing generalises to unseen
phrasings, while staying tiny enough (270M) to run locally alongside the STT, main LLM, and TTS
models without a GPU. FunctionGemma is Google's function-calling variant of Gemma, so it already
speaks the `<start_function_call>` grammar — fine-tuning just teaches it *this* assistant's eight
tools and, critically, when to stay silent.

## Method

- **LoRA fine-tune** (PEFT + TRL `SFTTrainer`), no quantization — plain `transformers`/`trl`/`peft`,
  no `unsloth`/`bitsandbytes`.
- LoRA config: `r=8`, `lora_alpha=16`, `lora_dropout=0.05`, `bias="none"`,
  `target_modules="all-linear"`.
- 8 epochs, `learning_rate=5e-5` (constant schedule), effective batch size 4
  (`per_device_train_batch_size=2 × gradient_accumulation_steps=2`), `max_length=700`,
  `optim="adamw_torch_fused"`.
- Config lives in [`configs/train.yaml`](configs/train.yaml); the training loop is
  [`train/sft_engine.py`](train/sft_engine.py).
- **~19 min total** (~2.7 min/epoch), derived from the epoch timestamps in
  `checkpoints/Jun19_16-36-02_032d8990b2f7.json`. Trained on a single CUDA GPU (`torch cu128`).

## The dataset is the interesting part

1,660 examples (1,328 train / 332 test), and roughly **a quarter of them are deliberately
negative**:

- **Positive (tool) examples** — hand-written natural-language prompts per tool in
  [`data/raw/`](data/raw), expanded by argument injection: `{internal_id}` placeholders are
  replaced with both digits and words ("one".."five") so the model learns arguments robustly
  ([`data_pipeline/generate_synthetic.py`](data_pipeline/generate_synthetic.py)).
- **Negative (no-tool) examples** — ~25% of the final set is `NO_TOOL`, drawn from
  `cosmosai471/General_Conversation_Mixed_Dataset`, so the model learns to **not** fire a tool
  during ordinary conversation. A router that calls a tool on every turn is useless; teaching it
  restraint is the whole point ([`data_pipeline/final_split.py`](data_pipeline/final_split.py)).
- **Schemas are generated from code**: each tool is a Python function in
  [`data_pipeline/schemas/tools.py`](data_pipeline/schemas/tools.py); its docstring is turned into
  a JSON schema by `transformers.utils.get_json_schema`, so the training schemas can never drift
  from the tool signatures.

The eight tools are the assistant's paper-reading domain: `get_papers`, `get_summary`,
`stage_paper`, `print_paper`, `list_titles`, `get_staged_id`, `get_time`, `get_date`.

## From adapter to production

The path from a LoRA checkpoint to a model Ollama can serve, including one non-obvious fix:

1. **Merge** the LoRA adapter into the base weights → `final_merged_export`
   ([`scripts/export_model.py`](scripts/export_model.py)).
2. **Sanitize the tokenizer** — the Gemma-3 tokenizer ships multimodal tokens
   (`<image_soft_token>`, `<end_of_image>`) whose IDs exceed the text-only model's vocab, which
   breaks GGUF conversion. [`scripts/ai/sanitize_gemma3_text_tokenizer.py`](scripts/ai/sanitize_gemma3_text_tokenizer.py)
   strips them.
3. **Convert to GGUF** (f16) via llama.cpp's `convert_hf_to_gguf.py`.
4. **Publish to Hugging Face** → [`MattWall100/functiongemma-alfredAI-merged-router-GGUF`](https://huggingface.co/MattWall100/functiongemma-alfredAI-merged-router-GGUF),
   consumed by the assistant as an Ollama Modelfile.

## Repository structure

```text
FunctionGemma-Finetune/
├── configs/train.yaml            # training hyperparameters
├── data/
│   ├── raw/                      # hand-written prompt templates per tool
│   ├── processed/                # argument-injected + NO_TOOL examples
│   ├── final/                    # final_train.jsonl (1328) / final_test.jsonl (332)
│   └── schemas.jsonl             # tool JSON schemas
├── data_pipeline/
│   ├── schemas/tools.py          # the 8 tools (docstrings = schema source)
│   ├── schemas/generate_schemas.py
│   ├── generate_synthetic.py     # argument injection + negative-example builder
│   └── final_split.py            # balance negatives + train/test split
├── train/
│   ├── sft_config.py             # Pydantic settings + YAML loader
│   └── sft_engine.py             # TRL SFTTrainer + LoRA wiring
├── scripts/
│   ├── run_training.py           # training entry point
│   ├── export_model.py           # merge LoRA adapter into base
│   └── ai/                       # tokenizer sanitize + HF upload helpers
└── tests/
    ├── evaluate.py               # tool-name accuracy eval
    ├── evaluate_exact_match.py   # exact-token-match eval
    └── lengths.py                # token-length statistics
```

## Reproducing the pipeline

Setup:

```bash
git clone ...
cd FunctionGemma-Finetune
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate data:

1. store tools in `tools.py`
2. Generate schemas:
   ```bash
   python .\data_pipeline\schemas\generate_schemas.py
   ```
3. Store synthetic data in `.\data\raw\tool_name.txt` line by line.
   If any argument is needed, replace in-line with `{argument_name}`.
   add a line to `[CONFIG LOCATION]` to specify per-tool argument variants that will be injected during data processing.
4. Create processed data with argument injection variants:
   ```bash
   python data_pipeline.generate_synthetic
   ```
5. Get final split:
   ```bash
   python -m data_pipeline.final_split
   ```

Train:

```bash
python -m scripts.run_training
```

Export — merge, sanitize the Gemma-3 tokenizer, convert to GGUF, upload to HF:

```bash
# Sanitize gemma3 tokenizer
python3 scripts.ai.sanitize_gemma3_text_tokenizer.py --model-dir final_merged_model

# Download llama.cpp repo to use convert_hf_to_gguf.py
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
python3 -m pip install -r requirements/requirements-convert_hf_to_gguf.txt

# Convert final_merged_export to GGUF and upload to HF
export HF_TOKEN=...
python -m scripts.upload_gguf_model \
 --repo-id your-username/your-model-name-GGUF \
 --converter /path/to/llama.cpp/convert_hf_to_gguf.py
```

Extras:

```bash
# $env:PYTHONUTF8 = "1"
hf auth login
--extra-index-url https://download.pytorch.org/whl/cu128

# Use convert_hf_to_gguf.py directly
python3 convert_hf_to_gguf.py /path/to/huggingface/model --outfile model.gguf

# Upload the merged HF model (no GGUF conversion)
export HF_TOKEN=...
python -m scripts.upload_merged_model --repo-id your-username/your-model-name

# Upload an already-converted GGUF
python -m scripts.upload_gguf_model \
 --repo-id your-username/your-model-name-GGUF \
 --outfile gguf/functiongemma-merged.f16.gguf \
 --skip-convert
```

## Known gaps / rough edges

- `configs/data_pipeline.yaml` is empty — the `[CONFIG LOCATION]` referenced in the data steps was
  never wired up, so argument injections are still hardcoded in `generate_synthetic.py` (only
  `internal_id` is defined).
- The two eval scripts score different checkpoints (`checkpoint-282` = epoch 1 vs
  `checkpoint-2256` = epoch 8) and both have the base-model baseline commented out, so reproducing
  the 63.5% → 99.5% comparison means re-enabling those lines. Checkpoints are gitignored.
- `scripts/export_model.py` hardcodes `checkpoint-282` and still has debug `print` lines.
