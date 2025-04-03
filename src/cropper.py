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
    

def cropper(X: np.ndarray, M: np.ndarray) -> list[np.ndarray]:
    """
    Crops the images based on the masks to produce the zetas (cropped objects).

    Args:
        X: np.ndarray, images with shape (n, height, width, channels).
        M: np.ndarray, binary masks with shape (n, height, width).

    Returns:
        zetas: list[np.ndarray], cropped images with background removed.
    """
    try:
        zetas = []
        
        # Ensure we have the same number of images and masks
        if X.shape[0] != M.shape[0]:
            raise ValueError(f"Number of images ({X.shape[0]}) must match number of masks ({M.shape[0]})")

        for i in range(X.shape[0]):
            image = X[i]  # Current image
            mask = M[i]   # Corresponding mask

            # Convert mask to binary (if stored in 255 format)
            if mask.max() == 255:
                mask = mask // 255  # Normalize to {0, 1}

            # Find the bounding box of the mask
            if np.any(mask):  # Ensure the mask is not empty
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)

                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]

                # Crop the image and mask
                cropped = image[y_min:y_max+1, x_min:x_max+1]
                cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]

                # Apply mask to remove background
                if len(cropped.shape) == 3:  # RGB image
                    cropped_masked = cropped * np.expand_dims(cropped_mask, axis=-1)
                else:  # Grayscale
                    cropped_masked = cropped * cropped_mask

                zetas.append(cropped_masked)
                print(f"Image {i} cropped to shape: {cropped_masked.shape}")
            else:
                print(f"Warning: Mask {i} is empty. Adding blank image.")
                zetas.append(np.zeros((10, 10, 3) if len(image.shape) == 3 else (10, 10), dtype=np.uint8))

        return zetas

    except Exception as e:
        print(f"Error cropping data: {e}")
        return []
