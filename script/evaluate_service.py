import requests
import os

API_URL = "http://localhost:8000/predict"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
TEST_IMAGE_DIR = os.path.join(PROJECT_ROOT, "test_images")

test_samples = [
    ("cat1.jpg", "cat"),
    ("dog1.jpg", "dog"),
    ("cat2.jpg", "cat"),
]

correct = 0

for filename, true_label in test_samples:
    image_path = os.path.join(TEST_IMAGE_DIR, filename)

    with open(image_path, "rb") as f:
        response = requests.post(API_URL, files={"file": f})
        pred_label = response.json()["label"]

    print(f"{filename} | Predicted: {pred_label} | True: {true_label}")

    if pred_label == true_label:
        correct += 1

accuracy = correct / len(test_samples)
print(f"\nPost-deployment Accuracy: {accuracy:.2f}")