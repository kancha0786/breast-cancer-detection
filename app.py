
"""
Breast Cancer Detection — Gradio Demo App
=========================================
Binary Classification: Benign vs Malignant
Models: EfficientNetB3 + DenseNet121 + ResNet50 (Weighted Ensemble)

Run:
    pip install -r requirements.txt
    python app.py
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import gradio as gr

import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ────────────────────────────────────────────────────────────────
# Paths (robust for GitHub / local execution)
# ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
EXAMPLES_DIR = BASE_DIR / "examples"

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Benign", "Malignant"]
MODEL_NAMES = ["efficientnet_b3", "densenet121", "resnet50"]

NUM_CLASSES = 2
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────
# Image preprocessing
# ────────────────────────────────────────────────────────────────
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])


# ────────────────────────────────────────────────────────────────
# Model Builder
# ────────────────────────────────────────────────────────────────
def build_model(name: str):
    model = timm.create_model(
        name,
        pretrained=False,
        num_classes=0,
        drop_rate=0.3
    )

    in_features = model.num_features

    classifier_head = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )

    if hasattr(model, "classifier"):
        model.classifier = classifier_head
    elif hasattr(model, "fc"):
        model.fc = classifier_head
    elif hasattr(model, "head"):
        model.head = classifier_head
    elif hasattr(model, "heads"):
        model.heads = classifier_head

    return model.to(DEVICE)


# ────────────────────────────────────────────────────────────────
# Load Models
# ────────────────────────────────────────────────────────────────
print("Loading models...")
models = {}

for name in MODEL_NAMES:
    weight_path = MODELS_DIR / f"{name}_best.pth"

    if weight_path.exists():
        model = build_model(name)

        state = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(state)

        model.eval()
        models[name] = model
        print(f"Loaded: {name}")

    else:
        print(f"Missing weights: {weight_path.name}")

if len(models) == 0:
    raise FileNotFoundError(
        "No model weights found inside /models folder."
    )


# ────────────────────────────────────────────────────────────────
# Load Ensemble Weights
# ────────────────────────────────────────────────────────────────
weights_file = MODELS_DIR / "ensemble_weights.json"

if weights_file.exists():
    with open(weights_file, "r") as f:
        ens_weights = json.load(f)

    print("Loaded ensemble weights.")

else:
    ens_weights = {name: 1 / len(models) for name in models}
    print("Using equal ensemble weights.")


print(f"Running on device: {DEVICE}")


# ────────────────────────────────────────────────────────────────
# Prediction Function
# ────────────────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return "No image uploaded.", {}, "Please upload an image."

    # PIL -> numpy
    img = np.array(image.convert("RGB"))

    # preprocess
    tensor = transform(image=img)["image"].unsqueeze(0).to(DEVICE)

    model_probs = {}
    model_outputs = {}

    with torch.no_grad():
        for name, model in models.items():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            model_probs[name] = probs

            pred_idx = int(np.argmax(probs))

            model_outputs[name] = {
                "label": CLASS_NAMES[pred_idx],
                "benign": float(probs[0]),
                "malignant": float(probs[1])
            }

    # weighted ensemble
    final_probs = np.zeros(NUM_CLASSES, dtype=np.float32)

    total_weight = sum(
        ens_weights.get(name, 1.0) for name in models
    )

    for name in models:
        weight = ens_weights.get(name, 1.0) / total_weight
        final_probs += weight * model_probs[name]

    pred_idx = int(np.argmax(final_probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(final_probs[pred_idx])

    # label output
    label_scores = {
        "Benign": float(final_probs[0]),
        "Malignant": float(final_probs[1])
    }

    # details markdown
    details = "## Per-model Predictions\n\n"
    details += "| Model | Prediction | Benign % | Malignant % |\n"
    details += "|------|------------|----------|-------------|\n"

    for name, result in model_outputs.items():
        details += (
            f"| {name} | "
            f"{result['label']} | "
            f"{result['benign']*100:.1f}% | "
            f"{result['malignant']*100:.1f}% |\n"
        )

    details += (
        f"\n## Final Ensemble Prediction: **{pred_label}**\n"
        f"Confidence: **{confidence*100:.2f}%**\n"
    )

    if pred_label == "Malignant":
        details += (
            "\n⚠️ Research output suggests malignant pattern. "
            "Consult a qualified medical professional."
        )
    else:
        details += (
            "\n✅ Research output suggests benign pattern. "
            "Consult a qualified medical professional."
        )

    return pred_label, label_scores, details


# ────────────────────────────────────────────────────────────────
# Example Images
# ────────────────────────────────────────────────────────────────
examples = []

for fname in [
    "benign_sample.png",
    "malignant_sample.png"
]:
    fpath = EXAMPLES_DIR / fname
    if fpath.exists():
        examples.append([str(fpath)])


# ────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Breast Cancer Detection",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
# 🔬 Breast Cancer Histopathology Detection

### Binary Classification using Deep Learning Ensemble

Upload a breast histopathology image and receive:

- Benign / Malignant prediction  
- Confidence scores  
- Per-model outputs  
- Ensemble decision  

> ⚠️ For research and educational use only.
> Not for clinical diagnosis.
""")

    with gr.Row():

        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Upload Histopathology Image",
                height=320
            )

            run_btn = gr.Button(
                "Analyse Image",
                variant="primary"
            )

        with gr.Column():
            text_output = gr.Textbox(
                label="Prediction",
                lines=1
            )

            label_output = gr.Label(
                label="Confidence Scores",
                num_top_classes=2
            )

    details_output = gr.Markdown()

    run_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[
            text_output,
            label_output,
            details_output
        ]
    )

    if examples:
        gr.Examples(
            examples=examples,
            inputs=image_input,
            label="Example Images"
        )

    gr.Markdown("""
---
### Project Summary

| Item | Value |
|------|------|
| Dataset | BreaKHis |
| Images | 7,909 |
| Patients | 82 |
| Models | EfficientNetB3, DenseNet121, ResNet50 |
| Method | Weighted Soft Voting |
| Framework | PyTorch + timm + Gradio |

Full notebook available on Kaggle.
""")


# ────────────────────────────────────────────────────────────────
# Launch
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_port=7860,
        show_error=True
    )"""
Breast Cancer Detection — Gradio Demo App
=========================================
Binary Classification: Benign vs Malignant
Models: EfficientNetB3 + DenseNet121 + ResNet50 (Weighted Ensemble)

Run:
    pip install -r requirements.txt
    python app.py
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import gradio as gr

import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ────────────────────────────────────────────────────────────────
# Paths (robust for GitHub / local execution)
# ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
EXAMPLES_DIR = BASE_DIR / "examples"

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Benign", "Malignant"]
MODEL_NAMES = ["efficientnet_b3", "densenet121", "resnet50"]

NUM_CLASSES = 2
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────
# Image preprocessing
# ────────────────────────────────────────────────────────────────
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])


# ────────────────────────────────────────────────────────────────
# Model Builder
# ────────────────────────────────────────────────────────────────
def build_model(name: str):
    model = timm.create_model(
        name,
        pretrained=False,
        num_classes=0,
        drop_rate=0.3
    )

    in_features = model.num_features

    classifier_head = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )

    if hasattr(model, "classifier"):
        model.classifier = classifier_head
    elif hasattr(model, "fc"):
        model.fc = classifier_head
    elif hasattr(model, "head"):
        model.head = classifier_head
    elif hasattr(model, "heads"):
        model.heads = classifier_head

    return model.to(DEVICE)


# ────────────────────────────────────────────────────────────────
# Load Models
# ────────────────────────────────────────────────────────────────
print("Loading models...")
models = {}

for name in MODEL_NAMES:
    weight_path = MODELS_DIR / f"{name}_best.pth"

    if weight_path.exists():
        model = build_model(name)

        state = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(state)

        model.eval()
        models[name] = model
        print(f"Loaded: {name}")

    else:
        print(f"Missing weights: {weight_path.name}")

if len(models) == 0:
    raise FileNotFoundError(
        "No model weights found inside /models folder."
    )


# ────────────────────────────────────────────────────────────────
# Load Ensemble Weights
# ────────────────────────────────────────────────────────────────
weights_file = MODELS_DIR / "ensemble_weights.json"

if weights_file.exists():
    with open(weights_file, "r") as f:
        ens_weights = json.load(f)

    print("Loaded ensemble weights.")

else:
    ens_weights = {name: 1 / len(models) for name in models}
    print("Using equal ensemble weights.")


print(f"Running on device: {DEVICE}")


# ────────────────────────────────────────────────────────────────
# Prediction Function
# ────────────────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return "No image uploaded.", {}, "Please upload an image."

    # PIL -> numpy
    img = np.array(image.convert("RGB"))

    # preprocess
    tensor = transform(image=img)["image"].unsqueeze(0).to(DEVICE)

    model_probs = {}
    model_outputs = {}

    with torch.no_grad():
        for name, model in models.items():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            model_probs[name] = probs

            pred_idx = int(np.argmax(probs))

            model_outputs[name] = {
                "label": CLASS_NAMES[pred_idx],
                "benign": float(probs[0]),
                "malignant": float(probs[1])
            }

    # weighted ensemble
    final_probs = np.zeros(NUM_CLASSES, dtype=np.float32)

    total_weight = sum(
        ens_weights.get(name, 1.0) for name in models
    )

    for name in models:
        weight = ens_weights.get(name, 1.0) / total_weight
        final_probs += weight * model_probs[name]

    pred_idx = int(np.argmax(final_probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(final_probs[pred_idx])

    # label output
    label_scores = {
        "Benign": float(final_probs[0]),
        "Malignant": float(final_probs[1])
    }

    # details markdown
    details = "## Per-model Predictions\n\n"
    details += "| Model | Prediction | Benign % | Malignant % |\n"
    details += "|------|------------|----------|-------------|\n"

    for name, result in model_outputs.items():
        details += (
            f"| {name} | "
            f"{result['label']} | "
            f"{result['benign']*100:.1f}% | "
            f"{result['malignant']*100:.1f}% |\n"
        )

    details += (
        f"\n## Final Ensemble Prediction: **{pred_label}**\n"
        f"Confidence: **{confidence*100:.2f}%**\n"
    )

    if pred_label == "Malignant":
        details += (
            "\n⚠️ Research output suggests malignant pattern. "
            "Consult a qualified medical professional."
        )
    else:
        details += (
            "\n✅ Research output suggests benign pattern. "
            "Consult a qualified medical professional."
        )

    return pred_label, label_scores, details


# ────────────────────────────────────────────────────────────────
# Example Images
# ────────────────────────────────────────────────────────────────
examples = []

for fname in [
    "benign_sample.png",
    "malignant_sample.png"
]:
    fpath = EXAMPLES_DIR / fname
    if fpath.exists():
        examples.append([str(fpath)])


# ────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Breast Cancer Detection",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
# 🔬 Breast Cancer Histopathology Detection

### Binary Classification using Deep Learning Ensemble

Upload a breast histopathology image and receive:

- Benign / Malignant prediction  
- Confidence scores  
- Per-model outputs  
- Ensemble decision  

> ⚠️ For research and educational use only.
> Not for clinical diagnosis.
""")

    with gr.Row():

        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Upload Histopathology Image",
                height=320
            )

            run_btn = gr.Button(
                "Analyse Image",
                variant="primary"
            )

        with gr.Column():
            text_output = gr.Textbox(
                label="Prediction",
                lines=1
            )

            label_output = gr.Label(
                label="Confidence Scores",
                num_top_classes=2
            )

    details_output = gr.Markdown()

    run_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[
            text_output,
            label_output,
            details_output
        ]
    )

    if examples:
        gr.Examples(
            examples=examples,
            inputs=image_input,
            label="Example Images"
        )

    gr.Markdown("""
---
### Project Summary

| Item | Value |
|------|------|
| Dataset | BreaKHis |
| Images | 7,909 |
| Patients | 82 |
| Models | EfficientNetB3, DenseNet121, ResNet50 |
| Method | Weighted Soft Voting |
| Framework | PyTorch + timm + Gradio |

Full notebook available on Kaggle.
""")


# ────────────────────────────────────────────────────────────────
# Launch
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_port=7860,
        show_error=True
    )




