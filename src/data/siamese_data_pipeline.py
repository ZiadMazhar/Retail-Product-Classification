import itertools
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

### locak import ###
# none

### type definitions  ###
DS_t = tuple[
    np.ndarray, np.ndarray
]  # (X, y): X.shape = (2, n, h, w, c), y.shape = (n,) where n is the number of examples, h is the height, w is the width, and c is the number of channels

### load up the env var ###
# none


def load_an_image(path: str, x_shape: tuple[int, int]) -> np.ndarray:
    """
    Load an image from a path, resize it, normalize pixel values,
    and convert it to the RGB color space.

    Args:
        path (str): The path to the image.
        x_shape (tuple[int, int]): The target (height, width) of the image.

    Returns:
        np.ndarray: Preprocessed image as a NumPy array.
    """
    try:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise ValueError(f"Failed to load image at {path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img_resized = cv2.resize(img_rgb, x_shape)  # Resize
        image_normalized = img_resized.astype("float32") / 255.0  # Normalize

        return image_normalized
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return np.zeros(
            (*x_shape, 3), dtype="float32"
        )  # Return blank image as fallback


def load_data(
    root_in_path: str, X_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load positive and negative images from a directory.

    Args:
        root_in_path (str): Path to the root directory containing 'P' and 'N' folders.
        X_shape (tuple[int, int]): Shape to resize the images to.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays containing positive and negative images.
    """
    try:

        def load_images_from_folder(folder: str) -> np.ndarray:
            images = []
            for filename in os.listdir(folder):
                img_path = os.path.join(folder, filename)
                img = load_an_image(img_path, X_shape)  # Use the fixed function
                images.append(img)
            return np.array(images, dtype="float32")

        X_positive = load_images_from_folder(os.path.join(root_in_path, "P"))
        X_negative = load_images_from_folder(os.path.join(root_in_path, "N"))

        return X_positive, X_negative
    except Exception as e:
        print(f"Error loading data: {e}")
        return np.empty((0, *X_shape, 3), dtype="float32"), np.empty(
            (0, *X_shape, 3), dtype="float32"
        )


def augment_data(
    X_t: np.ndarray, num_augmented: int, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment the data by applying 4 transformations to the images
    args:
        X: np.ndarray, the data to augment

        num_augmented: int, the number of times to augment the data
    returns:
        X_augmented: tuple[np.ndarray, np.ndarray] , the augmented data
    """
    data_augmentation = keras.Sequential(
        [
            layers.RandomZoom(0.2),
            layers.RandomRotation(0.2),
            layers.RandomFlip("horizontal"),
            layers.RandomContrast(0.2),
        ]
    )
    try:
        X_augmented = []
        Y_augmented = []
        if verbose:
            print(f"Applying data augmentation to {len(X_t)} images...")
        for i in range(num_augmented):
            X_augmented.extend(
                data_augmentation(tf.expand_dims(x, axis=0))[0].numpy() for x in X_t[0]
            )
            Y_augmented.extend(
                data_augmentation(tf.expand_dims(x, axis=0))[0].numpy() for x in X_t[1]
            )

        if verbose:
            print(f"Data augmentation complete. Augmented {len(X_augmented)} images.")
        return np.array(X_augmented), np.array(Y_augmented)

    except Exception as e:
        print(f"Error augmenting data: {e}")
        return np.empty((0, *X_t[0].shape[1:]), dtype="float32"), np.empty((0, *X_t[1].shape[1:]), dtype="float32")


def generate_dataset_for_training(
    X_t: tuple[np.ndarray, np.ndarray], verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the dataset for training by
    creating the positive and negative pairs
    args:
        X: tuple[np.ndarray, np.ndarray], the data to generate the pairs from
    returns:
        X_t: tuple[np.ndarray, np.ndarray] , the training data
    """
    try:
        p, n = X_t  # Unpacking positive and negative examples
        # Generate all possible (positive, negative) pairs
        n_pairs = np.array(list(itertools.product(p, n)))
        if verbose:
            print(f"Generated {len(n_pairs)} positive-negative pairs.")
        # Generate all possible (positive, positive) pairs
        p_pairs = np.array(list(itertools.product(p, p)))
        if verbose:
            print(f"Generated {len(p_pairs)} positive-positive pairs.")

        return p_pairs, n_pairs  # Correctly return the tuple
    except Exception as e:
        print(f"Error generating dataset for training: {e}")
        return np.empty((0, *p.shape[1:]), dtype="float32"), np.empty((0, *n.shape[1:]), dtype="float32")


def data_pipeline(
    root_in_path: str,
    X_shape: tuple,
    num_augment: int,
    train_ratio: float,
    verbose: bool = False,
) -> tuple[DS_t, DS_t]:
    """
    main data pipeline to generate the dataset for training.
    args:
        root_in_path: str, path to the root directory
        X_shape: tuple, shape to resize the images to
        num_augment: int, number of augmentations to do
        train_ratio: float, ratio to split the data
        verbose: bool, whether to show the progress bar
    returns:
        (X_train, y_train), (X_test, y_test): tuple[DS_t, DS_t], the training and testing datasets
    """
    try:
        # Load data
        X_t = load_data(root_in_path, X_shape)

        # Augment data
        X_aug = augment_data(X_t, num_augment, verbose)

        # Combine original and augmented images properly
        X_t = (
            np.concatenate([X_t[0], X_aug[0]]),  # Positive images
            np.concatenate([X_t[1], X_aug[1]]),  # Negative images
        )

        # Generate positive-negative pairs
        X_p, X_n = generate_dataset_for_training(X_t, verbose)

        # Ensure X_p and X_n have compatible dimensions for concatenation
        min_size = min(X_p.shape[0], X_n.shape[0])
        X_p = X_p[:min_size]
        X_n = X_n[:min_size]

        # Assign labels
        y_p = np.array([1] * X_p.shape[0])  # (n,)
        y_n = np.array([0] * X_n.shape[0])  # (n,)

        # Stack positive and negative pairs along axis=0
        X_pairs = np.concatenate([X_p, X_n], axis=0)  # Shape: (2n, h, w, c)
        y_labels = np.concatenate([y_p, y_n])  # Shape: (2n,)

        # Shuffle along the n dimension
        indices = np.random.permutation(y_labels.shape[0])  # Shuffle indices for pairs
        X_pairs = X_pairs[indices]  # Shuffle along axis=0
        y_labels = y_labels[indices]  # Shuffle labels accordingly

        # Split into training and testing sets
        split_index = int(len(y_labels) * train_ratio)
        X_train, X_test = X_pairs[:split_index], X_pairs[split_index:]
        y_train, y_test = y_labels[:split_index], y_labels[split_index:]
        if verbose:
            print(
                f"Data pipeline complete. Training set: {len(y_train)}, Testing set: {len(y_test)}"
            )
        return [[X_train, y_train], [X_test, y_test]]
    except Exception as e:
        print(f"Error in data pipeline: {e}")
