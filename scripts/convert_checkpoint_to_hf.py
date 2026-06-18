"""Convert a pickle .pt training checkpoint to a HF-safe safetensors model.

Training checkpoints (scripts/train_mdn_asp.py) are saved with torch.save as
Python pickle: {"step", "model": state_dict, ["val_loss"]}.  Pickle is flagged
"unsafe" by the Hugging Face scanner because it can execute arbitrary code on
load.  This script reconstructs the MDNAsp module from the state dict and
re-serialises it with PyTorchModelHubMixin.save_pretrained, which writes
model.safetensors + config.json (no pickle, no banner).

Usage
-----
    python scripts/convert_checkpoint_to_hf.py \
        --ckpt results/mdn_asp_v2/best.pt \
        --out_dir results/mdn_asp_v2/hf

Optionally push to the Hub (requires `huggingface-cli login` first):
        --push_to_hub --repo_id <user>/mad-clean-mdn-asp

The config args must match the architecture the checkpoint was trained with.
The defaults below match scripts/train_mdn_asp.py defaults.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from mad_clean.models.mdn_asp import MDNAsp


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Convert .pt checkpoint to HF safetensors.")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to the pickle .pt checkpoint.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory for model.safetensors + config.json.")
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--hidden",        type=int, default=256)
    p.add_argument("--n_components",  type=int, default=5)
    p.add_argument("--cond_dim",      type=int, default=5)
    p.add_argument("--push_to_hub", action="store_true",
                   help="Push the converted model to the Hub (needs login).")
    p.add_argument("--repo_id", type=str, default=None,
                   help="Hub repo id, e.g. <user>/mad-clean-mdn-asp.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # weights_only=True is safe here: we only need the tensor state dict, and
    # it refuses to unpickle arbitrary objects.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))

    model = MDNAsp(
        base_channels=args.base_channels,
        hidden=args.hidden,
        n_components=args.n_components,
        cond_dim=args.cond_dim,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"State dict mismatch — architecture args likely wrong.\n"
            f"  missing:    {missing}\n  unexpected: {unexpected}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"[convert] Wrote safetensors model + config.json to {out_dir}/")

    if args.push_to_hub:
        if not args.repo_id:
            raise SystemExit("--push_to_hub requires --repo_id")
        model.push_to_hub(args.repo_id)
        print(f"[convert] Pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
