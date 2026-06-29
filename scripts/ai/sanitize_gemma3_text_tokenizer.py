import argparse
import json
import shutil
from pathlib import Path


OUT_OF_RANGE_IMAGE_TOKENS = {
    "<image_soft_token>",
    "<end_of_image>",
}

TOKENIZER_CONFIG_KEYS = {
    "image_token",
    "eoi_token",
}

MODEL_SPECIFIC_KEYS = {
    "image_token",
    "eoi_token",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def maybe_backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)


def sanitize_tokenizer_json(path: Path, vocab_size: int) -> list[tuple[str, int]]:
    data = load_json(path)
    removed: list[tuple[str, int]] = []
    kept_added_tokens = []

    for token in data.get("added_tokens", []):
        content = token.get("content")
        token_id = token.get("id")
        if (
            content in OUT_OF_RANGE_IMAGE_TOKENS
            and isinstance(token_id, int)
            and token_id >= vocab_size
        ):
            removed.append((content, token_id))
            continue
        kept_added_tokens.append(token)

    data["added_tokens"] = kept_added_tokens
    save_json(path, data)
    return removed


def sanitize_tokenizer_config(path: Path) -> list[str]:
    data = load_json(path)
    removed: list[str] = []

    for key in TOKENIZER_CONFIG_KEYS:
        if key in data:
            removed.append(key)
            data.pop(key)

    model_specific = data.get("model_specific_special_tokens")
    if isinstance(model_specific, dict):
        for key in MODEL_SPECIFIC_KEYS:
            if key in model_specific:
                removed.append(f"model_specific_special_tokens.{key}")
                model_specific.pop(key)

    save_json(path, data)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove Gemma 3 multimodal tokenizer entries that sit outside a "
            "text-only FunctionGemma embedding matrix."
        )
    )
    parser.add_argument(
        "--model-dir",
        default="final_merged_export",
        help="Merged Hugging Face export directory to sanitize.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak copies before editing tokenizer files.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    config_path = model_dir / "config.json"
    tokenizer_json_path = model_dir / "tokenizer.json"
    tokenizer_config_path = model_dir / "tokenizer_config.json"

    for path in (config_path, tokenizer_json_path, tokenizer_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = load_json(config_path)
    vocab_size = config.get("vocab_size")
    if not isinstance(vocab_size, int):
        raise ValueError(f"Expected integer vocab_size in {config_path}")

    if not args.no_backup:
        maybe_backup(tokenizer_json_path)
        maybe_backup(tokenizer_config_path)

    removed_tokens = sanitize_tokenizer_json(tokenizer_json_path, vocab_size)
    removed_config_keys = sanitize_tokenizer_config(tokenizer_config_path)

    print(f"Sanitized {model_dir}")
    print(f"vocab_size: {vocab_size}")
    print(f"removed added tokens: {removed_tokens}")
    print(f"removed tokenizer config keys: {removed_config_keys}")


if __name__ == "__main__":
    main()
