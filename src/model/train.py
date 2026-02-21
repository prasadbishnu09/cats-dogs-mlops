# src/model/train.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.pytorch
from pathlib import Path
from src.data.dataset import CatsDogsDataset
from model.net import SimpleCNN
import itertools
import random
import time
import os


def plot_loss(train_losses, val_losses, outpath):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(outpath)
    plt.close()


def plot_confusion(cm, classes, outpath):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(outpath)
    plt.close()


def train(args):
    print("Starting training script...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    run_name = args.run_name or f"run_{int(time.time())}"
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed
        })

        torch.manual_seed(args.seed)
        random.seed(args.seed)

        print("Loading datasets...")

        train_ds = CatsDogsDataset(args.processed, split="train", augment=True)
        val_ds = CatsDogsDataset(args.processed, split="val", augment=False)

        print("Train samples:", len(train_ds))
        print("Val samples:", len(val_ds))

        if len(train_ds) == 0:
            raise ValueError("Training dataset is empty!")

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0  # Windows-safe
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        model = SimpleCNN(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_acc = 0.0
        train_losses, val_losses = [], []
        patience = args.patience
        no_improve = 0

        print("Beginning training loop...")

        for epoch in range(args.epochs):
            epoch_start = time.time()

            model.train()
            running_loss = 0.0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * imgs.size(0)

            epoch_train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)

            # Validation
            model.eval()
            all_preds, all_labels = [], []
            val_loss = 0.0

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)

                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())

            epoch_val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
            val_acc = accuracy_score(all_labels, all_preds)

            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"train_loss={epoch_train_loss:.4f} | "
                f"val_loss={epoch_val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"time={time.time() - epoch_start:.1f}s"
            )

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0

                model_path = Path(args.model_out)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(str(model_path), artifact_path="model")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping triggered")
                    break

        # Artifacts
        plot_loss(train_losses, val_losses, "loss_curve.png")
        mlflow.log_artifact("loss_curve.png")

        try:
            cm = confusion_matrix(all_labels, all_preds)
            plot_confusion(cm, ["cat", "dog"], "confusion.png")
            mlflow.log_artifact("confusion.png")
        except Exception as e:
            print("Confusion matrix failed:", e)

        try:
            mlflow.pytorch.log_model(model, "pytorch_model")
        except Exception as e:
            print("MLflow model logging failed:", e)

        print("Training complete. Best val acc:", best_val_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=str, default="data/processed")
    parser.add_argument("--model_out", type=str, default="models/model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)  # smaller default for CPU
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--mlflow_experiment", type=str, default="cats-dogs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)
