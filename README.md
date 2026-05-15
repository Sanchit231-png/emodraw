# EmoDraw

An emotion-based drawing system that uses a Convolutional Neural Network (CNN) to classify emotions from images and maps them to visual drawing outputs. The project combines computer vision and deep learning to create an interactive creative tool driven by detected emotional state.

---

## How It Works

1. **Model Training** (`train_model.py`) — A CNN is trained on a labelled image dataset of emotional expressions. The model learns to distinguish between emotion categories and saves the best-performing weights.
2. **Emotion-Driven Drawing** (`emodraw.py`) — The trained model is used at inference time to classify input images and trigger corresponding drawing behaviour.

---

## Model Architecture

The CNN is built with Keras and follows a standard feature extraction + classification pattern:

```
Input (64×64 grayscale)
  → Conv2D (32 filters, 3×3, ReLU)
  → MaxPooling2D (2×2)
  → Conv2D (64 filters, 3×3, ReLU)
  → MaxPooling2D (2×2)
  → Flatten
  → Dropout (0.3)
  → Dense (128, ReLU)
  → Dense (N classes, Softmax)
```

**Training details:**
- Optimiser: Adam
- Loss: Categorical Cross-Entropy
- Epochs: 15, Batch size: 16
- 80/20 train/test split
- Best model saved via `ModelCheckpoint` (monitored on `val_accuracy`)

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| Python 3 | Core language |
| Keras / TensorFlow | CNN model definition and training |
| OpenCV (`cv2`) | Image loading and preprocessing |
| NumPy | Array operations |
| scikit-learn | Train/test split |

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare your dataset
# Place emotion-labelled images under: dataset/<emotion_label>/image.jpg

# Train the model
python train_model.py
# Saves best weights to: model/bestmodel.keras

# Run the drawing system
python emodraw.py
```

---

## Dataset Structure

```
dataset/
├── happy/
│   ├── img1.jpg
│   └── ...
├── sad/
│   └── ...
└── angry/
    └── ...
```

Each subdirectory name becomes the class label. Images are resized to 64×64 grayscale before training.

---

## Author

**Sanchit Mukherjee** — [LinkedIn](https://www.linkedin.com/in/sanchit-m-872819262/) · [GitHub](https://github.com/Sanchit231-png)
