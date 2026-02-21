# src/eval_online.py
import os
from pathlib import Path
from src.serve.utils import ModelWrapper
from PIL import Image
import random
from sklearn.metrics import accuracy_score
import mlflow

def simulate_and_eval(model_path, labeled_dir, sample_size=20):
    mw = ModelWrapper(model_path)
    items = []
    for lbl in ["cat","dog"]:
        p = Path(labeled_dir) / lbl
        if p.exists():
            for f in p.glob("*"):
                items.append((str(f), lbl))
    random.shuffle(items)
    items = items[:sample_size]
    trues, preds = [], []
    for f, t in items:
        b = open(f, "rb").read()
        res = mw.predict_from_bytes(b)
        preds.append(res["label"])
        trues.append(t)
    acc = accuracy_score(trues, preds)
    print(f"Simulated batch accuracy: {acc:.4f}")
    try:
        mlflow.log_metric("postdeploy_simulated_accuracy", acc)
    except Exception:
        pass
    return acc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.pt")
    parser.add_argument("--labeled", default="data/processed/224x224/val")
    args = parser.parse_args()
    simulate_and_eval(args.model, args.labeled)