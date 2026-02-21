# src/data/dataset.py
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class CatsDogsDataset(Dataset):
    def __init__(self, root_dir, split="train", size=224, augment=False):
        self.root_dir = Path(root_dir) / "224x224" / split
        self.classes = ["cat","dog"]
        self.paths = []
        for i, c in enumerate(self.classes):
            p = self.root_dir / c
            if p.exists():
                for f in p.glob("*"):
                    if f.suffix.lower() in {".jpg",".jpeg",".png"}:
                        self.paths.append((str(f), i))
        self.augment = augment
        base = [T.Resize((size,size)), T.ToTensor()]
        if augment:
            aug = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ])
            self.transform = T.Compose([aug] + base + [T.Normalize(mean=[0.485,0.456,0.406],
                                                                  std=[0.229,0.224,0.225])])
        else:
            self.transform = T.Compose(base + [T.Normalize(mean=[0.485,0.456,0.406],
                                                          std=[0.229,0.224,0.225])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p, label = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
