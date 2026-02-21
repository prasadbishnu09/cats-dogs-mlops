# src/tests/test_inference.py
import tempfile
from PIL import Image
import torch
from src.serve.utils import ModelWrapper
from src.model.net import SimpleCNN
from pathlib import Path

def test_model_predict_on_dummy_image(tmp_path):
    # create a dummy model file
    model = SimpleCNN()
    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)
    mw = ModelWrapper(str(model_path))
    img = Image.new("RGB", (224,224), (100,100,200))
    import io
    b = io.BytesIO()
    img.save(b, format="JPEG")
    res = mw.predict_from_bytes(b.getvalue())
    assert "label" in res and "probability" in res
    assert 0.0 <= res["probability"] <= 1.0