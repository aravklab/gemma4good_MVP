"""
scripts/download_whisper_model.py
==================================
One-time setup script: downloads a faster-whisper model from HuggingFace and
saves it to ./whisper_model/<model_name>/ inside the project directory.

Downloads files directly via HTTP (bypassing huggingface_hub's hf-xet engine)
so it works on corporate networks with SSL-inspection proxies.

USAGE
-----
    # From the Gemma4good/ directory:
    python scripts/download_whisper_model.py

    # Download a different model size:
    python scripts/download_whisper_model.py --model small

AVAILABLE MODELS
----------------
    tiny        ~75 MB   fast, less accurate
    base        ~145 MB  default, good balance         <- default
    small       ~460 MB  more accurate, slower
    medium      ~1.5 GB  high accuracy
    large-v3    ~3.0 GB  best accuracy, requires GPU or patience on CPU

To swap the model used by the app:
  1. Re-run this script with --model <size>
  2. Edit WHISPER_MODEL_PATH in app.py to point to ./whisper_model/<size>/
"""

import argparse
import os
import ssl
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# SSL bypass — must happen before any network call
# Disables certificate verification at the Python ssl level, which works
# regardless of which HTTP library is used underneath.
# ---------------------------------------------------------------------------
ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001

HF_BASE = "https://huggingface.co/Systran/faster-whisper-{model}/resolve/main/{file}"

# Files required by faster-whisper for every model size
MODEL_FILES = [
    "config.json",
    "tokenizer.json",
    "vocabulary.json",
    "model.bin",        # main weights (~75 MB tiny / ~145 MB base / etc.)
    "preprocessor_config.json",
]


def _download_file(url: str, dest: Path) -> None:
    """Stream a single file from url to dest with a progress bar."""
    req = urllib.request.Request(url, headers={"User-Agent": "python"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1 << 16  # 64 KB
        with open(dest, "wb") as fh:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                fh.write(block)
                downloaded += len(block)
                if total:
                    pct = downloaded * 100 // total
                    bar = "#" * (pct // 5)
                    print(
                        f"\r  [{bar:<20}] {pct:3d}%  "
                        f"({downloaded/1e6:.1f}/{total/1e6:.1f} MB)",
                        end="",
                        flush=True,
                    )
    print()  # newline after progress bar


def download(model_name: str) -> None:
    target_dir = Path(__file__).parent.parent / "whisper_model" / model_name

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Model '{model_name}' already present at:\n  {target_dir}")
        print("Delete the folder and re-run to force a fresh download.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading faster-whisper-{model_name} to:\n  {target_dir}\n")

    failed = []
    for filename in MODEL_FILES:
        url  = HF_BASE.format(model=model_name, file=filename)
        dest = target_dir / filename
        print(f"  {filename}")
        try:
            _download_file(url, dest)
        except Exception as exc:
            err = str(exc)
            if "404" in err or "Not Found" in err:
                # Some models omit optional files — skip silently
                dest.unlink(missing_ok=True)
                print(f"  (not present for this model size, skipping)")
            else:
                print(f"  ERROR: {exc}")
                dest.unlink(missing_ok=True)
                failed.append(filename)

    if failed:
        print(f"\nFailed to download: {failed}")
        print("If you are behind a corporate proxy, try connecting via VPN and re-running.")
        sys.exit(1)

    # Verify the essential weight file landed
    weight_file = target_dir / "model.bin"
    if not weight_file.exists() or weight_file.stat().st_size < 1_000_000:
        print("\nERROR: model.bin is missing or truncated — download may have been interrupted.")
        print("Delete the folder and re-run:")
        print(f"  python scripts/download_whisper_model.py --model {model_name}")
        sys.exit(1)

    print(f"\nDone. Model saved to: {target_dir}")
    print("app.py will load from this path automatically on the next run.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download a faster-whisper model locally.")
    p.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Model size to download (default: base).",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    download(args.model)
