#!/usr/bin/env python3
"""
Offline Model Pre-Downloader
================================

Run this ONCE on a machine with internet access to pre-cache the
ResNet50 weights. After that, the Car Audit Tool can run completely
offline (no internet required at runtime).

Usage:
    python scripts/download_model.py [--weights IMAGENET1K_V1|IMAGENET1K_V2]

The weights are cached under TORCH_HOME (default: ~/.cache/torch).
Set TORCH_HOME env var to change the cache location, e.g.:

    TORCH_HOME=./model_cache python scripts/download_model.py
"""

import argparse
import os
import sys
import hashlib

import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights


WEIGHT_OPTIONS = {
    "IMAGENET1K_V1": ResNet50_Weights.IMAGENET1K_V1,
    "IMAGENET1K_V2": ResNet50_Weights.IMAGENET1K_V2,
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-download ResNet50 weights for offline use.")
    parser.add_argument(
        "--weights",
        choices=list(WEIGHT_OPTIONS.keys()),
        default="IMAGENET1K_V2",
        help="Which pre-trained weight variant to cache (default: IMAGENET1K_V2).",
    )
    args = parser.parse_args()

    torch_home = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
    hub_dir = os.path.join(torch_home, "hub", "checkpoints")
    os.makedirs(hub_dir, exist_ok=True)

    print("=" * 60)
    print("Car Audit Tool - Offline Model Pre-Downloader")
    print("=" * 60)
    print(f"TORCH_HOME     : {torch_home}")
    print(f"Cache dir      : {hub_dir}")
    print(f"Weight variant : {args.weights}")
    print("=" * 60)

    weight_enum = WEIGHT_OPTIONS[args.weights]
    print("\nDownloading weights (this may take a few minutes the first time)...")
    try:
        model = models.resnet50(weights=weight_enum)
    except Exception as e:
        print(f"\nERROR: Failed to download weights: {e}", file=sys.stderr)
        print("Make sure you have internet access on this machine.", file=sys.stderr)
        return 1

    # Trigger a forward pass to JIT-load any lazy submodules
    dummy = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        _ = model(dummy)

    # Locate the cached .pth file
    cached_files = [
        f for f in os.listdir(hub_dir)
        if f.endswith(".pth")
    ]

    print("\n" + "=" * 60)
    print("SUCCESS - Cached weight files:")
    print("=" * 60)
    if not cached_files:
        print(f"  (no .pth files found in {hub_dir} - check torch hub behavior)")
    for f in cached_files:
        full = os.path.join(hub_dir, f)
        size_mb = os.path.getsize(full) / (1024 * 1024)
        print(f"  {full}  ({size_mb:.2f} MB)")
        print(f"    sha256: {sha256_file(full)}")

    print("\nYou can now run the Car Audit Tool completely offline.")
    print("  python app.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
