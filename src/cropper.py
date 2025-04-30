import numpy as np
import tensorflow as tf
import os

### local imports ###
#none

### type definitions ###    
#none   

def load_X_and_M(X_root_path: str, M_root_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the images and masks from the given paths.

    Args:
        X_root_path: str, path to the images.
        M_root_path: str, path to the masks.

    Returns:
        X: np.ndarray, images.
        M: np.ndarray, masks.
    """
    try:
        X_list = []
        M_list = []
        
        # Load images
        for root, _, files in os.walk(X_root_path):
            for file in files:
                img_path = os.path.join(root, file)
                img = tf.keras.preprocessing.image.load_img(img_path)
                img = np.array(img, dtype=np.float32)  
                X_list.append(img)
        
        # Load masks
        for root, _, files in os.walk(M_root_path):
            for file in files:
                mask_path = os.path.join(root, file)
                mask = tf.keras.preprocessing.image.load_img(mask_path, color_mode="grayscale")
                mask = np.array(mask, dtype=np.uint8)  
                M_list.append(mask)
        
        # Convert lists to NumPy arrays
        X = np.array(X_list) if X_list else np.empty((0, 512, 512, 3), dtype=np.float32)
        M = np.array(M_list) if M_list else np.empty((0, 512, 512 ), dtype=np.uint8)

        return X, M

    except Exception as e:
        print(f"Error loading data: {e}")
        return np.empty((0, 512, 512, 3), dtype="float32"), np.empty((0, 512, 512, 3), dtype="float32")
    
def cropper(image: np.ndarray, bboxes: np.ndarray) -> list[np.ndarray]:
    """
    Crops the image based on the provided bounding boxes.

    Args:
        image: np.ndarray, the input image (H, W, C).
        bboxes: np.ndarray, array of bounding boxes with shape (N, 4), each as [x_min, y_min, x_max, y_max].

    Returns:
        crops: list[np.ndarray], list of cropped images.
    """
    crops = []
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError("bboxes must be of shape (N, 4)")
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox.astype(int)
        crop = image[y_min:y_max+1, x_min:x_max+1]
        crops.append(crop)
    return crops
