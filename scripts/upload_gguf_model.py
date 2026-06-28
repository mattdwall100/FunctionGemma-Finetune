import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_MODEL_DIR = "final_merged_export"
DEFAULT_OUTFILE = "gguf/functiongemma-merged.f16.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the merged Hugging Face export to GGUF with llama.cpp, then "
            "upload the GGUF file to a Hugging Face model repo."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_GGUF_REPO_ID") or os.environ.get("HF_REPO_ID"),
        help="Destination model repo, for example username/functiongemma-merged-GGUF.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Merged Hugging Face model directory. Defaults to {DEFAULT_MODEL_DIR}.",
    )
    parser.add_argument(
        "--outfile",
        default=DEFAULT_OUTFILE,
        help=f"Local GGUF output path. Defaults to {DEFAULT_OUTFILE}.",
    )
    parser.add_argument(
        "--converter",
        default=os.environ.get("LLAMA_CPP_CONVERT_SCRIPT"),
        help=(
            "Path to llama.cpp/convert_hf_to_gguf.py. Can also be set with "
            "LLAMA_CPP_CONVERT_SCRIPT."
        ),
    )
    parser.add_argument(
        "--outtype",
        default="f16",
        choices=["f32", "f16", "bf16", "q8_0", "auto"],
        help="GGUF output tensor type passed to llama.cpp conversion.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Upload --outfile as-is without running conversion first.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Filename/path to use inside the Hugging Face repo. Defaults to outfile name.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the destination repo as private if it does not exist.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload FunctionGemma GGUF export",
        help="Commit message for the Hub upload.",
    )
    return parser.parse_args()


def require_existing_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Merged model directory does not exist: {path}")

    required_files = ["config.json", "model.safetensors", "tokenizer_config.json"]
    missing_files = [name for name in required_files if not (path / name).exists()]
    if missing_files:
        joined = ", ".join(missing_files)
        raise FileNotFoundError(f"Merged model directory is missing required files: {joined}")


def convert_to_gguf(
    converter: Path,
    model_dir: Path,
    outfile: Path,
    outtype: str,
) -> None:
    if not converter.is_file():
        raise FileNotFoundError(
            "Could not find llama.cpp converter script. Pass --converter or set "
            f"LLAMA_CPP_CONVERT_SCRIPT. Received: {converter}"
        )

    outfile.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(converter),
        str(model_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if not args.repo_id:
        raise ValueError("Set --repo-id, HF_GGUF_REPO_ID, or HF_REPO_ID.")

    model_dir = Path(args.model_dir).resolve()
    outfile = Path(args.outfile).resolve()
    require_existing_dir(model_dir)

    if not args.skip_convert:
        if not args.converter:
            raise ValueError("Pass --converter or set LLAMA_CPP_CONVERT_SCRIPT.")
        convert_to_gguf(
            converter=Path(args.converter).resolve(),
            model_dir=model_dir,
            outfile=outfile,
            outtype=args.outtype,
        )

    if not outfile.is_file():
        raise FileNotFoundError(f"GGUF file does not exist: {outfile}")

    path_in_repo = args.path_in_repo or outfile.name

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=str(outfile),
        path_in_repo=path_in_repo,
        commit_message=args.commit_message,
    )

    print(f"Uploaded {outfile} to https://huggingface.co/{args.repo_id}/blob/main/{path_in_repo}")


if __name__ == "__main__":
    main()
