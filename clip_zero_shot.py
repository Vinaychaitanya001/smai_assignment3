"""
T7.5 — CLIP Zero-Shot Wildlife Classifier
==========================================
Uses openai/clip-vit-base-patch32 (no training!) to classify
6 Indian wildlife species: cheetah, fox, hyena, lion, tiger, wolf.

Outputs:
  - Classification report (precision, recall, F1)
  - Confusion matrix plot  → results/confusion_matrix.png
  - Per-class accuracy bar  → results/per_class_accuracy.png
  - Top-3 accuracy metric
"""

import os, json, time, warnings
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
import torch
from transformers import CLIPProcessor, CLIPModel

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────
DATASET_DIR  = Path("dataset")
RESULTS_DIR  = Path("results")
RESOLUTION   = "224"                       # use the 224-px images
MODEL_NAME   = "openai/clip-vit-base-patch32"

CLASSES = ["cheetah", "fox", "hyena", "lion", "tiger", "wolf"]

# text prompts — richer descriptions help CLIP
TEXT_PROMPTS = [
    "a photo of a cheetah, a large spotted African cat",
    "a photo of a fox, a small reddish canine with a bushy tail",
    "a photo of a hyena, an African carnivore with rounded ears",
    "a photo of a lion, a large maned African cat",
    "a photo of a tiger, a large striped Asian cat",
    "a photo of a gray wolf, a large wild canine",
]

RESULTS_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────
def gather_images():
    """Return list of (image_path, label_index) for every valid image in the dataset."""
    samples = []
    skipped = 0
    for idx, cls_name in enumerate(CLASSES):
        folder_name = f"{cls_name}-resize-{RESOLUTION}"
        cls_dir = DATASET_DIR / folder_name
        # handle nested structure (some folders have an extra subfolder)
        for root, _dirs, files in os.walk(cls_dir):
            for f in sorted(files):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    fpath = os.path.join(root, f)
                    # validate image can be opened
                    try:
                        img = Image.open(fpath)
                        img.verify()  # check for corruption
                        samples.append((fpath, idx))
                    except Exception:
                        skipped += 1
                        print(f"  [WARN] Skipping corrupted image: {fpath}")
    if skipped:
        print(f"  Skipped {skipped} corrupted image(s)")
    return samples


def load_model():
    """Load CLIP model + processor."""
    print(f"Loading {MODEL_NAME} …")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def classify_batch(model, processor, image_paths, batch_size=32):
    """
    Classify a list of image paths using CLIP zero-shot.
    Returns (top1_preds, top3_preds, all_probs) as numpy arrays.
    """
    text_inputs = processor(text=TEXT_PROMPTS, return_tensors="pt", padding=True)

    all_probs, all_top1, all_top3 = [], [], []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                # create a blank placeholder for corrupted files
                images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
        image_inputs = processor(images=images, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(
                **{k: v for k, v in image_inputs.items()},
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            logits = outputs.logits_per_image          # (B, num_classes)
            probs  = logits.softmax(dim=-1).cpu().numpy()

        top1 = probs.argmax(axis=1)
        top3 = probs.argsort(axis=1)[:, -3:][:, ::-1]  # descending

        all_probs.append(probs)
        all_top1.extend(top1.tolist())
        all_top3.extend(top3.tolist())

        print(f"  Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)}", end="\r")

    print()
    return np.array(all_top1), np.array(all_top3), np.vstack(all_probs)


# ── Plotting ──────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("CLIP Zero-Shot — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved {RESULTS_DIR / 'confusion_matrix.png'}")


def plot_per_class_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#FF6B6B", "#FFA94D", "#FFD43B", "#69DB7C", "#4DABF7", "#9775FA"]
    bars = ax.bar(CLASSES, per_class_acc * 100, color=colors, edgecolor="white", linewidth=1.5)
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc*100:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("CLIP Zero-Shot — Per-Class Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_class_accuracy.png", dpi=150)
    plt.close()
    print(f"  Saved {RESULTS_DIR / 'per_class_accuracy.png'}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("T7.5 — CLIP Zero-Shot Wildlife Classifier")
    print("=" * 60)

    # 1. Gather images
    samples = gather_images()
    print(f"\nTotal images: {len(samples)}")
    for i, cls in enumerate(CLASSES):
        n = sum(1 for _, lbl in samples if lbl == i)
        print(f"  {cls:>10s}: {n}")

    image_paths = [p for p, _ in samples]
    y_true      = np.array([lbl for _, lbl in samples])

    # 2. Load model
    model, processor = load_model()

    # 3. Classify
    print("\nRunning zero-shot classification …")
    t0 = time.time()
    top1_preds, top3_preds, probs = classify_batch(model, processor, image_paths, batch_size=16)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  ({len(samples)/elapsed:.1f} img/s)")

    # 4. Metrics
    top1_acc = accuracy_score(y_true, top1_preds)
    top3_acc = np.mean([y_true[i] in top3_preds[i] for i in range(len(y_true))])
    print(f"\n{'Top-1 Accuracy':>20s}: {top1_acc*100:.2f}%")
    print(f"{'Top-3 Accuracy':>20s}: {top3_acc*100:.2f}%")

    print("\nClassification Report:")
    report = classification_report(y_true, top1_preds, target_names=CLASSES, digits=3)
    print(report)

    # Save report to file
    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write("T7.5 — CLIP Zero-Shot Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Total images: {len(samples)}\n")
        f.write(f"Top-1 Accuracy: {top1_acc*100:.2f}%\n")
        f.write(f"Top-3 Accuracy: {top3_acc*100:.2f}%\n\n")
        f.write(report)
    print(f"  Saved {RESULTS_DIR / 'classification_report.txt'}")

    # 5. Plots
    print("\nGenerating plots …")
    plot_confusion_matrix(y_true, top1_preds)
    plot_per_class_accuracy(y_true, top1_preds)

    print("\n[DONE] All done!")


if __name__ == "__main__":
    main()
