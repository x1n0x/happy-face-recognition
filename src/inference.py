from pathlib import Path

import numpy as np
from PIL import Image


def preprocess_image(image_path, image_size=(48, 48)):
    """
    Load and preprocess a single image for CNN inference.

    Steps:
    - convert to grayscale
    - resize to fixed resolution
    - normalize pixel values to [0, 1]
    - add channel and batch dimensions
    """
    img = Image.open(image_path).convert("L")
    img = img.resize(image_size)

    img = np.asarray(img, dtype=np.float32) / 255.0
    img = img[..., np.newaxis]        # channel dimension
    img = np.expand_dims(img, axis=0) # batch dimension

    return img


def predict_image(model, image_path, threshold=0.5):
    """
    Run inference on a single image and return probability and label.
    """
    img_input = preprocess_image(image_path)
    prob = model.predict(img_input, verbose=0)[0][0]
    label = "Happy" if prob >= threshold else "Non-happy"

    return prob, label


def predict_folder(model, image_dir, threshold=0.5):
    """
    Run inference on all images in a directory.

    Only files with common image extensions are considered.
    Each image is processed independently.
    """
    image_dir = Path(image_dir)

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    results = []

    for img_path in image_paths:
        prob, label = predict_image(model, img_path, threshold)
        results.append({
            "path": img_path,
            "probability": prob,
            "label": label
        })

    return results
