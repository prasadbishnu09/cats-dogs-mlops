# src/data/preprocess.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import shutil

def resize_and_save(src_path, dest_path, size=(224,224)):
    img = Image.open(src_path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest_path, format="JPEG", quality=90)

def build_splits(raw_root, out_root, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    random.seed(seed)
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    labels = ["dog", "cat"]
    for lbl in labels:
        src_dir = Path(raw_root) / lbl
        files = [p for p in src_dir.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]
        for subset, fset in [("train", train_files), ("val", val_files), ("test", test_files)]:
            for p in tqdm(fset, desc=f"{lbl}-{subset}"):
                rel = p.name
                dest = Path(out_root) / "224x224" / subset / lbl / rel
                resize_and_save(p, dest)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="data/raw/dogs-vs-cats")
    parser.add_argument("--out", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_splits(args.raw, args.out, seed=args.seed)

if __name__ == "__main__":
    main()
