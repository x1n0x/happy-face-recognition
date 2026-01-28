from pathlib import Path

import numpy as np
from PIL import Image


POSITIVE_CLASS = "happy"


def get_label_from_class(class_name: str) -> int:
    """
    Convert class name to binary label.
    happy -> 1
    non-happy -> 0
    """
    return 1 if class_name == POSITIVE_CLASS else 0


def load_image(path: Path, image_size=(48, 48)) -> np.ndarray:
    """
    Load an image, convert to grayscale, resize, and normalize.
    """
    img = Image.open(path).convert("L")
    img = img.resize(image_size)
    img = np.asarray(img, dtype=np.float32) / 255.0
    return img


def load_dataset(base_dir: Path):
    """
    Load images and labels from a directory structured by class folders.
    """
    images = []
    labels = []

    base_dir = Path(base_dir)

    for class_dir in sorted(base_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        label = get_label_from_class(class_dir.name)

        for img_path in class_dir.glob("*"):
            img = load_image(img_path)
            images.append(img)
            labels.append(label)

    X = np.stack(images)
    y = np.array(labels, dtype=np.int64)

    return X, y


def load_train_test(data_dir: Path):
    """
    Load train and test splits using the same preprocessing pipeline.
    """
    data_dir = Path(data_dir)

    X_train, y_train = load_dataset(data_dir / "train")
    X_test, y_test = load_dataset(data_dir / "test")

    return X_train, y_train, X_test, y_test
