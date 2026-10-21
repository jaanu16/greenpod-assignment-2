
# Greenpod Labs — Assignment 2
## Computer Vision: Fruit Condition Classification

This project implements and compares two computer vision approaches
for classifying orange-condition images into four classes:

- Blackspot
- Canker
- Fresh
- Grenning

### Objectives

1. Explore and preprocess the fruit image dataset.
2. Extract handcrafted color and texture features.
3. Train an SVM classifier.
4. Fine-tune a MobileNetV2 classification head using transfer learning.
5. Compare both models using accuracy, precision, recall, and F1-score.
6. Demonstrate predictions on individual images.

> Important: The current experiment uses orange-condition images.
> The original assignment specifies avocado ripening-stage
> classification. These are different tasks.

### Dataset

- Total images: 1,090
- Training images: 991
- Test images: 99
- Classes: blackspot, canker, fresh, grenning

The test set contains 22 blackspot, 22 canker, 33 fresh,
and 22 grenning images.

Dataset access:
https://drive.google.com/file/d/1NMJ0FOxcWH1mwFFeERuY6fgEY-MVgeET/view

Verify that the linked dataset corresponds to the orange dataset
before using it. Keep downloaded data out of Git unless you have
permission to redistribute it.

### Methods

#### Approach 1: Classical machine learning

- Resize images to 128 x 128.
- Extract HSV color histograms.
- Extract Local Binary Pattern (LBP) texture features.
- Standardize features.
- Train an SVM with an RBF kernel.

#### Approach 2: Deep learning

- Use ImageNet-pretrained MobileNetV2.
- Resize images to 224 x 224.
- Apply training-time augmentation.
- Normalize images using ImageNet statistics.
- Train a four-class classification head.

### Reported experimental results

| Metric | SVM | MobileNetV2 |
|---|---:|---:|
| Accuracy | 97.98% | 93.94% |
| Macro precision | 97.73% | 93.80% |
| Macro recall | 97.73% | 93.18% |
| Macro F1-score | 97.73% | 93.42% |

These are the results supplied in the project report. Run the
provided code and verify that the results match your saved
predictions before claiming they were reproduced.

### Repository structure

- `notebooks/`: exploratory analysis, training, and evaluation.
- `src/`: reusable Python implementation.
- `scripts/`: command-line entry points.
- `models/`: saved model artifacts.
- `results/`: metrics, plots, and predictions.
- `report/`: final assignment report.
- `video/`: walkthrough recording information.

### Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd greenpod-assignment-2

python -m venv .venv
```

Activate the environment:

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Dataset setup

Arrange the images in this format:

```text
data/raw/
├── blackspot/
├── canker/
├── fresh/
└── grenning/
```

Each class directory should contain its corresponding images.

### Running the project

Inspect the dataset:

```bash
python scripts/check_dataset.py
```

Train the classical model:

```bash
python -m src.train_classical
```

Train MobileNetV2:

```bash
python -m src.train_deep_learning
```

Evaluate saved models and generate a demo prediction:

```bash
python scripts/run_demo.py path/to/image.jpg
```

### Evaluation

The evaluation pipeline should report:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Classification report
- Confusion matrix

Use validation data for model selection and reserve the test
set for final evaluation.

### Limitations

- The test set contains only 99 images.
- Lighting, backgrounds, and fruit appearance can introduce bias.
- Duplicate images across splits may cause data leakage.
- Softmax scores are not necessarily calibrated probabilities.
- Inference latency must be benchmarked on the target hardware.

### Original assignment requirement

To complete the original avocado task, repeat the experiment
with the avocado dataset and its four ripening stages:
underripe, breaking, ripe, and overripe.

### Links

- GitHub: ADD_YOUR_REPOSITORY_LINK
- Google Colab: ADD_YOUR_COLAB_LINK
- Walkthrough video: ADD_YOUR_VIDEO_LINK
- Final report: `report/Assignment2_Report.pdf`
