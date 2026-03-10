#!/usr/bin/env python3
"""Download the LLM model from HuggingFace Hub."""

import argparse
import sys

from huggingface_hub import snapshot_download


RECOMMENDED_MODELS = {
    "32b-fp8": "LGAI-EXAONE/EXAONE-4.0-32B-FP8",    # 33GB, fits H200 easily
    "32b": "LGAI-EXAONE/EXAONE-4.0.1-32B",            # 64GB, fits H200
    "7.8b": "LGAI-EXAONE/EXAONE-Deep-7.8B",           # 16GB, lightweight
    "k-exaone": "LGAI-EXAONE/K-EXAONE-236B-A23B",     # 474GB, needs 4xH200
    "k-exaone-fp8": "LGAI-EXAONE/K-EXAONE-236B-A23B-FP8",  # 245GB, needs 2xH200
}


def main():
    parser = argparse.ArgumentParser(description="Download EXAONE model for BDS")
    parser.add_argument(
        "model",
        nargs="?",
        default="32b-fp8",
        help=f"Model alias or full HF ID. Aliases: {list(RECOMMENDED_MODELS.keys())}",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Custom cache directory (default: HF default cache)",
    )
    args = parser.parse_args()

    model_id = RECOMMENDED_MODELS.get(args.model, args.model)
    print(f"Downloading: {model_id}")
    print("This may take a while depending on model size and network speed...")

    try:
        path = snapshot_download(
            model_id,
            cache_dir=args.cache_dir,
            resume_download=True,
        )
        print(f"\nModel downloaded to: {path}")
        print(f"\nTo serve: MODEL={model_id} bash scripts/serve_model.sh")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
