#!/usr/bin/env python3
"""Download public Vietnamese ASR datasets from Hugging Face.

Usage:
    python scripts/download_data.py --dataset bud500 --out data/bud500
    python scripts/download_data.py --dataset vivos --out data/vivos
"""
import argparse
from pathlib import Path

from datasets import load_dataset

DATASETS = {
    "bud500": "linhtran92/viet_bud500",
    "vivos": "vivos",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--split", default=None, help="Optional split filter, e.g. 'train'"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_id = DATASETS[args.dataset]
    print(f"Loading {hf_id} from Hugging Face (this can take a while for large sets)...")
    ds = load_dataset(hf_id, split=args.split)

    print(f"Saving to {out_dir} (Arrow format, loadable with datasets.load_from_disk)...")
    ds.save_to_disk(str(out_dir))
    print("Done.")
    print(f"Example row keys: {list(ds[0].keys()) if len(ds) else 'empty dataset'}")


if __name__ == "__main__":
    main()
