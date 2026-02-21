# src/serve/utils.py
import io
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
from prometheus_client import Counter, Histogram

# Prometheus metrics (these names will be registered once)
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("inference_request_latency_seconds", "Latency for inference requests", ["endpoint"])
PREDICTION_COUNTER = Counter("predictions_total", "Total predictions by label", ["label"])

class ModelWrapper:
    def __init__(self, model_path, device=None):
        import torch
        from src.model.net import SimpleCNN
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN()
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.labels = ["cat","dog"]

    def predict_from_bytes(self, image_bytes):
        REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
        with REQUEST_LATENCY.labels("/predict").time():
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            x = self.transform(img).unsqueeze(0)
            import torch
            x = x.to(self.device)
            with torch.no_grad():
                out = self.model(x)
                prob = torch.softmax(out, dim=1).cpu().numpy()[0]
                idx = int(prob.argmax())
                label = self.labels[idx]
                PREDICTION_COUNTER.labels(label=label).inc()
                return {"label": label, "probability": float(prob[idx])}