
import argparse
import json
import random
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

CLASS_NAMES = ["blackspot", "canker", "fresh", "grenning"]
IMAGE_SIZE = 224
CLASSICAL_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 3
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_directories():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Dataset inspection
# --------------------------------------------------

def inspect_dataset(data_dir):
    """Print class counts and save a class distribution plot."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}"
        )

    print("\nDATASET CLASS DISTRIBUTION")
    counts = {}

    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

        count = sum(
            1 for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }
        )
        counts[class_name] = count
        print(f"{class_name:12s}: {count}")

    plt.figure(figsize=(8, 4))
    plt.bar(counts.keys(), counts.values(), color="forestgreen")
    plt.title("Fruit Condition Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "class_distribution.png")
    plt.close()

    return counts


# --------------------------------------------------
# Approach 1: Classical CV + SVM
# --------------------------------------------------

def extract_features(image_path):
    """Extract HSV color histograms and LBP texture features."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(
        image, (CLASSICAL_SIZE, CLASSICAL_SIZE)
    )

    # HSV color features
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_features = []

    for channel, bins, limits in [
        (0, 32, [0, 180]),
        (1, 32, [0, 256]),
        (2, 32, [0, 256]),
    ]:
        hist = cv2.calcHist(
            [hsv], [channel], None, [bins], limits
        )
        hist = cv2.normalize(hist, hist).flatten()
        color_features.extend(hist)

    # LBP texture features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(
        gray, P=8, R=1, method="uniform"
    )

    texture_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        range=(0, 10),
        density=True,
    )

    return np.concatenate(
        [np.asarray(color_features), texture_hist]
    ).astype(np.float32)


def load_classical_data(data_dir):
    features, labels = [], []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }:
                continue

            try:
                features.append(extract_features(image_path))
                labels.append(label)
            except (ValueError, cv2.error) as exc:
                print(f"Skipping {image_path}: {exc}")

    if not features:
        raise ValueError("No readable images found.")

    return np.asarray(features), np.asarray(labels)


def train_classical(data_dir):
    print("\nTraining classical SVM...")

    X, y = load_classical_data(data_dir)

    # The input folders are assumed to contain training data.
    # Use a separate held-out test directory for final evaluation.
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale"),
    )

    model.fit(X, y)

    model_path = MODEL_DIR / "svm_model.joblib"
    joblib.dump(model, model_path)

    print(f"SVM saved to {model_path}")
    return model


# --------------------------------------------------
# Approach 2: MobileNetV2
# --------------------------------------------------

def get_transforms(training=False):
    operations = []

    if training:
        operations.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
            ),
        ])

    operations.extend([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(operations)


def build_model(num_classes):
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Could not load pretrained MobileNetV2 weights. "
            "Check your internet connection or torchvision setup."
        ) from exc

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )

    return model


def train_deep_learning(data_dir, epochs=EPOCHS):
    print("\nTraining MobileNetV2...")
    print(f"Device: {DEVICE}")

    dataset = datasets.ImageFolder(
        str(data_dir),
        transform=get_transforms(training=True),
    )

    if dataset.classes != CLASS_NAMES:
        raise ValueError(
            f"Expected classes {CLASS_NAMES}, "
            f"but ImageFolder found {dataset.classes}"
        )

    if len(dataset) < 2:
        raise ValueError("At least two images are required.")

    # Recreate the split deterministically.
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # Validation data must not use random augmentation.
    val_base = datasets.ImageFolder(
        str(data_dir),
        transform=get_transforms(training=False),
    )
    val_ds.dataset = val_base

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    model = build_model(len(CLASS_NAMES)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=0.001
    )

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)
        val_loss /= len(val_ds)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_weights is not None:
        model.load_state_dict(best_weights)

    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
    }

    model_path = MODEL_DIR / "mobilenetv2_model.pth"
    torch.save(checkpoint, model_path)

    print(f"MobileNetV2 saved to {model_path}")
    return model.to(DEVICE)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def report_metrics(y_true, y_pred, class_names, prefix):
    print(f"\n{prefix} EVALUATION")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(
        "Precision: "
        f"{precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )
    print(
        "Recall:    "
        f"{recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )
    print(
        "Macro F1:  "
        f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )

    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    ))

    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names)))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    display.plot(cmap="Greens", xticks_rotation=30)
    plt.title(f"{prefix} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{prefix.lower()}_confusion_matrix.png")
    plt.close()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def evaluate_classical(model, test_dir):
    X, y = load_classical_data(test_dir)
    predictions = model.predict(X)

    return report_metrics(
        y, predictions, CLASS_NAMES, "SVM"
    )


def evaluate_deep_learning(model, test_dir):
    test_dataset = datasets.ImageFolder(
        str(test_dir),
        transform=get_transforms(training=False),
    )

    if test_dataset.classes != CLASS_NAMES:
        raise ValueError(
            f"Expected classes {CLASS_NAMES}, "
            f"but found {test_dataset.classes}"
        )

    loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(predictions)

    return report_metrics(
        np.asarray(y_true), np.asarray(y_pred),
        CLASS_NAMES, "MobileNetV2"
    )


# --------------------------------------------------
# Single-image prediction
# --------------------------------------------------

def predict_image(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    svm_path = MODEL_DIR / "svm_model.joblib"
    cnn_path = MODEL_DIR / "mobilenetv2_model.pth"

    if not svm_path.exists() or not cnn_path.exists():
        raise FileNotFoundError(
            "Train both models first using --mode train."
        )

    svm = joblib.load(svm_path)
    svm_prediction = int(
        svm.predict(extract_features(image_path).reshape(1, -1))[0]
    )

    checkpoint = torch.load(
        cnn_path, map_location=DEVICE, weights_only=True
    )

    class_names = checkpoint["class_names"]
    cnn = build_model(len(class_names))
    cnn.load_state_dict(checkpoint["state_dict"])
    cnn = cnn.to(DEVICE).eval()

    image = Image.open(image_path).convert("RGB")
    tensor = get_transforms(training=False)(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = cnn(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        cnn_prediction = int(probabilities.argmax().item())

    print("\nPREDICTION RESULTS")
    print(f"Image: {image_path}")
    print(f"SVM prediction: {class_names[svm_prediction]}")
    print(f"MobileNetV2 prediction: {class_names[cnn_prediction]}")
    print(
        "MobileNetV2 softmax score: "
        f"{probabilities[cnn_prediction].item() * 100:.2f}%"
    )


# --------------------------------------------------
# Main entry point
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Greenpod Labs fruit condition classifier"
    )

    parser.add_argument(
        "--mode",
        choices=["inspect", "train", "evaluate", "predict"],
        default="inspect",
    )
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--test-data", type=Path, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)

    args = parser.parse_args()

    set_seed()
    prepare_directories()

    if args.mode == "inspect":
        inspect_dataset(args.data)

    elif args.mode == "train":
        inspect_dataset(args.data)
        train_classical(args.data)
        train_deep_learning(args.data, epochs=args.epochs)

    elif args.mode == "evaluate":
        if args.test_data is None:
            parser.error(
                "--test-data is required for evaluation"
            )

        svm = joblib.load(MODEL_DIR / "svm_model.joblib")
        checkpoint = torch.load(
            MODEL_DIR / "mobilenetv2_model.pth",
            map_location=DEVICE,
            weights_only=True,
        )

        cnn = build_model(len(checkpoint["class_names"]))
        cnn.load_state_dict(checkpoint["state_dict"])
        cnn = cnn.to(DEVICE)

        results = {
            "SVM": evaluate_classical(svm, args.test_data),
            "MobileNetV2": evaluate_deep_learning(
                cnn, args.test_data
            ),
        }

        with open(RESULT_DIR / "metrics.json", "w") as file:
            json.dump(results, file, indent=4)

        print(f"\nMetrics saved to {RESULT_DIR / 'metrics.json'}")

    elif args.mode == "predict":
        if args.image is None:
            parser.error("--image is required for prediction")

        predict_image(args.image)


if __name__ == "__main__":
    main()
