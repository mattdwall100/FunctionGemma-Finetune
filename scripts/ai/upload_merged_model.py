import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_MODEL_DIR = "final_merged_export"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the merged Hugging Face model directory as-is."
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_REPO_ID"),
        help="Destination model repo, for example username/functiongemma-merged.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Directory to upload. Defaults to {DEFAULT_MODEL_DIR}.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the destination repo as private if it does not exist.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload merged FunctionGemma export",
        help="Commit message for the Hub upload.",
    )
    return parser.parse_args()


def require_existing_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {path}")

    required_files = ["config.json", "model.safetensors", "tokenizer_config.json"]
    missing_files = [name for name in required_files if not (path / name).exists()]
    if missing_files:
        joined = ", ".join(missing_files)
        raise FileNotFoundError(f"Model directory is missing required files: {joined}")


def main() -> None:
    args = parse_args()
    if not args.repo_id:
        raise ValueError("Set --repo-id or HF_REPO_ID.")

    model_dir = Path(args.model_dir).resolve()
    require_existing_dir(model_dir)

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message=args.commit_message,
    )

    print(f"Uploaded {model_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
