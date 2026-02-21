# src/tests/test_preprocess.py

import tempfile
from pathlib import Path
from PIL import Image
import shutil
from src.data.preprocess import resize_and_save


def test_resize_and_save_creates_image():
    tmpdir = Path(tempfile.mkdtemp())

    src = tmpdir / "src.jpg"
    img = Image.new("RGB", (500, 500), (255, 0, 0))
    img.save(src)

    dest = tmpdir / "out" / "img.jpg"
    resize_and_save(src, dest, size=(224, 224))

    assert dest.exists()

    # Properly close file before deleting directory
    with Image.open(dest) as im:
        assert im.size == (224, 224)

    shutil.rmtree(tmpdir)