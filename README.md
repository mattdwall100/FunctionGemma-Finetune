Finetuning Function Gemma to act as the sole tool-routing mechanism in local-AI-assistant pipeline.

Skeleton:

function-gemma-finetune/
├── .gitignore
├── README.md
├── requirements.txt # Explicit dependency pinning
├── config/ # Configuration-driven design
│ ├── model_config.yaml # Hardware, paths, and model variables
│ └── hyperparams.yaml # Learning rates, batch sizes, epochs
├── data/ # Strictly structured data tracking
│ ├── raw/ # Untouched base datasets
│ ├── processed/ # Cleaned/combined intermediate formats
│ └── final/ # Ready-to-train datasets (JSONL)
├── src/ # Core codebase package
│ ├── **init**.py
│ ├── schemas/
│ │ ├── **init**.py
│ │ ├── tools.py # Code-first definitions of your tools
│ │ └── generate_schemas.py # Script that exports python tools to JSON schema
│ ├── data/
│ │ ├── **init**.py
│ │ ├── generate_synthetic.py # Synthetic dataset generation logic
│ │ └── dataloader.py # Custom dataset maps and chat template formatters
│ ├── training/
│ │ ├── **init**.py
│ │ └── train.py # SFTTrainer orchestration script
│ └── utils/
│ ├── **init**.py
│ └── helpers.py # Logging setups, memory trackers
└── tests/ # Automated validation framework
├── **init**.py
├── test_data.py # Unit tests for schemas and templates
├── test_hardware.py # Hardware dry-run profiling (OOM/Speed checking)
└── run_hparam_sweep.sh # Orchestration bash script for parameter validation
