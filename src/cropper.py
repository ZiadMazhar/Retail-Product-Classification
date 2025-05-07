import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
### local imports ###
#none

### type definitions ###    
#none       
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
    
    # Plot original image with bboxes in different colors
    import matplotlib.pyplot as plt
    import random

    plt.figure(figsize=(8, 8))
    plt.imshow(image.astype(np.uint8))
    ax = plt.gca()
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox.astype(int)
        color = [random.random() for _ in range(3)]
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                             linewidth=2, edgecolor=color, facecolor='none', label=f"BBox {i}")
        ax.add_patch(rect)
    plt.title("Original Image with BBoxes")
    plt.axis('off')
    plt.show()

    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox.astype(int)
        crop = image[y_min:y_max+1, x_min:x_max+1]
        crops.append(crop)

    # # visualize the crops
    # batch_size = 20
    # if crops:
    #     for batch_start in range(0, len(crops), batch_size):
    #         batch = crops[batch_start:batch_start + batch_size]
    #         plt.figure(figsize=(3 * len(batch), 3))
    #         for idx, crop in enumerate(batch):
    #             plt.subplot(1, len(batch), idx + 1)
    #             plt.imshow(crop.astype(np.uint8))
    #             plt.title(f"Crop {batch_start + idx}")
    #             plt.axis('off')
    #         plt.tight_layout()
    #         plt.show()

    return crops
