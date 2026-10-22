  """
Avocado/Orange Fruit Condition Classification
----------------------------------------------
This script converts the Google Colab notebook into a regular Python program.

Expected dataset structure:
    Avocado Dataset.zip
or an extracted folder such as:
    dataset/
        train/
            blackspot/
            canker/
            fresh/
            grenning/
        test/
            blackspot/
            canker/
            fresh/
            grenning/

Run:
    python main.py --data-zip "Avocado Dataset.zip"
or:
    python main.py --data-dir dataset

The script trains and evaluates:
1. SVM using HSV color histograms + Local Binary Pattern texture features
2. MobileNetV2 transfer-learning model

Models and class labels are saved to the models/ directory.
"""

import argparse
import json
import random
import zipfile
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["blackspot", "canker", "fresh", "grenning"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def prepare_dataset(data_dir: Path, data_zip: Path | None) -> Path:
    """Extract a dataset ZIP if supplied, otherwise use the dataset directory."""
    if data_zip:
        if not data_zip.is_file() or not zipfile.is_zipfile(data_zip):
            raise FileNotFoundError(f"Dataset ZIP not found or invalid: {data_zip}")
        data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(data_zip, "r") as archive:
            archive.extractall(data_dir)
        print(f"Dataset extracted to: {data_dir}")

    # Support a ZIP that contains either train/ directly or one parent folder.
    if (data_dir / "train").is_dir():
        return data_dir

    candidates = [p for p in data_dir.rglob("train") if p.is_dir()]
    if candidates:
        return candidates[0].parent

    raise FileNotFoundError(
        f"Could not find a train/ folder inside {data_dir}. "
        "Check the dataset folder structure."
    )


def extract_features(image_path: Path) -> np.ndarray | None:
    """Extract normalized HSV histograms and LBP texture features."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    image = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))

    features = np.concatenate([h_hist, s_hist, v_hist, lbp_hist]).astype(np.float32)
    return features / (features.sum() + 1e-8)


def load_svm_split(data_dir: Path, split: str):
    """Load image features and labels for the SVM model."""
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset split: {split_dir}")

    features, labels = [], []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            feature = extract_features(image_path)
            if feature is not None:
                features.append(feature)
                labels.append(class_dir.name)

    if not features:
        raise ValueError(f"No readable images found in {split_dir}")

    return np.asarray(features), np.asarray(labels)


class FruitDataset(Dataset):
    """Image dataset for MobileNetV2."""

    def __init__(self, root: Path, class_to_idx: dict, transform=None):
        self.root = root
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = []

        if not root.is_dir():
            raise FileNotFoundError(f"Missing dataset split: {root}")

        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if class_dir.name not in class_to_idx:
                continue
            for path in class_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, class_to_idx[class_dir.name]))

        if not self.samples:
            raise ValueError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def train_svm(data_dir: Path, model_dir: Path):
    print("\n=== SVM: HSV + LBP features ===")
    x_train, y_train = load_svm_split(data_dir, "train")
    x_test, y_test = load_svm_split(data_dir, "test")

    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")
    print(f"Feature count: {x_train.shape[1]}")

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, class_weight="balanced"),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nSVM test accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, xticks_rotation=45, cmap="Blues"
    )
    plt.title("SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(model_dir / "svm_confusion_matrix.png", dpi=150)
    plt.close()

    joblib.dump(model, model_dir / "orange_svm_model.joblib")
    print(f"SVM model saved to: {model_dir / 'orange_svm_model.joblib'}")
    return model, {"Model": "SVM (HSV + LBP)", "Accuracy": accuracy,
                   "Macro Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
                   "Macro Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
                   "Macro F1": f1_score(y_test, predictions, average="macro", zero_division=0)}


def train_mobilenet(data_dir: Path, model_dir: Path, epochs: int, batch_size: int):
    print("\n=== MobileNetV2 transfer learning ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_root, test_root = data_dir / "train", data_dir / "test"
    class_names = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if not class_names:
        raise ValueError(f"No class folders found in {train_root}")
    class_to_idx = {name: index for index, name in enumerate(class_names)}

    train_ds = FruitDataset(train_root, class_to_idx, train_transform)
    test_ds = FruitDataset(test_root, class_to_idx, test_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print("Classes:", class_names)
    print(f"Training images: {len(train_ds)}")
    print(f"Testing images: {len(test_ds)}")

    # Pretrained weights require internet on the first run.
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {running_loss / max(total, 1):.4f} | "
            f"Training accuracy: {correct / max(total, 1):.4f}"
        )

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
            y_true.extend(labels.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nMobileNetV2 test accuracy: {accuracy:.4f}")
    print(classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names,
        xticks_rotation=45, cmap="Greens"
    )
    plt.title("MobileNetV2 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(model_dir / "mobilenet_confusion_matrix.png", dpi=150)
    plt.close()

    torch.save(model.state_dict(), model_dir / "orange_mobilenetv2.pth")
    with open(model_dir / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=4)
    print(f"MobileNetV2 weights saved to: {model_dir / 'orange_mobilenetv2.pth'}")

    metrics = {
        "Model": "MobileNetV2",
        "Accuracy": accuracy,
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    return model, class_names, test_transform, device, metrics


def main():
    parser = argparse.ArgumentParser(description="Train fruit-condition classification models.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"),
                        help="Directory to extract/use the dataset (default: dataset)")
    parser.add_argument("--data-zip", type=Path, default=None,
                        help="Optional path to the dataset ZIP file")
    parser.add_argument("--model-dir", type=Path, default=Path("models"),
                        help="Directory to save trained models and plots")
    parser.add_argument("--epochs", type=int, default=3,
                        help="MobileNetV2 training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size (default: 32)")
    parser.add_argument("--skip-mobilenet", action="store_true",
                        help="Train only the SVM model")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    data_dir = prepare_dataset(args.data_dir, args.data_zip)

    _, svm_metrics = train_svm(data_dir, args.model_dir)
    results = [svm_metrics]

    if not args.skip_mobilenet:
        _, _, _, _, mobilenet_metrics = train_mobilenet(
            data_dir, args.model_dir, args.epochs, args.batch_size
        )
        results.append(mobilenet_metrics)

    comparison = pd.DataFrame(results)
    print("\n=== Model comparison ===")
    print(comparison.round(4).to_string(index=False))
    comparison.to_csv(args.model_dir / "model_comparison.csv", index=False)
    print(f"\nAll outputs saved in: {args.model_dir.resolve()}")


if __name__ == "__main__":
    main()
