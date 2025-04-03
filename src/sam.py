import numpy as np
import tensorflow as tf
import os
from typing import Dict, Any, Tuple, List, Union, Optional



### local imports ###
from data.sam_helper import SamHelper


### type definitions ###
# none 
### load up the env var ###
# none


def main(
    img_name: str = "img1.png", 
    do_generate: bool = True,
    points: List[Tuple[int, int]] = None,
    point_labels: List[int] = None,
    box: List[int] = None,
    model_type: str = "vit_b",
    checkpoint_path: str = "models/sam_vit_b.pth",
    return_results: bool = False
) -> Union[int, Dict[str, Any]]:
    """
    Main function for the SAM model that either generates or loads the masks for a given image
    
    Args:
        img_name: the name of the image to load
        do_generate: whether or not to generate the masks
        points: list of (x, y) coordinates for point prompts, defaults to [(100, 100), (200, 200)]
        point_labels: list of labels for points (1 for foreground, 0 for background), defaults to [1, 0]
        box: bounding box in format [x1, y1, x2, y2], used if points is None
        model_type: SAM model type (vit_b, vit_l, vit_h)
        checkpoint_path: path to the model checkpoint
        return_results: whether to return the results dictionary instead of status code
        
    Returns:
        If return_results is False: 0 if successful, -1 if failed
        If return_results is True: Dictionary with masks, scores, logits, and segmented image
    """
    try:
        # Check if image file exists
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image file not found: {img_name}")
        sam_helper_instance = SamHelper()
        # Initialize the SAM model
        sam_helper_instance.init_model(model_type=model_type, checkpoint_path=checkpoint_path)
        
        # Load the image
        image = tf.keras.preprocessing.image.load_img(
            img_name, color_mode="rgb", target_size=(512, 512)
        )
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.array(image, dtype=np.uint8)

        # Set the image for the predictor
        sam_helper_instance.set_image(image)

        # Generate the masks
        if do_generate:
            # Set default points if none provided
            if points is None and box is None:
                points = [(100, 100), (200, 200)]
                point_labels = [1, 0]
            
            # Generate masks based on provided prompts
            if points is not None:
                masks, scores, logits = sam_helper_instance.predict_masks_from_points(
                    points=points,
                    point_labels=point_labels,
                    multimask_output=True,
                )
            elif box is not None:
                masks, scores, logits = sam_helper_instance.predict_masks_from_box(
                    box=box,
                    multimask_output=True,
                )
            else:
                raise ValueError("Either points or box must be provided")
            
            # Visualize all masks
            SamHelper.visualize_masks(image, masks, scores)
            
            # Extract and visualize the best mask
            best_mask, best_score = SamHelper.extract_best_mask(masks, scores)
            SamHelper.visualize_masks(image, np.expand_dims(best_mask, axis=0), np.array([best_score]))
            
            # Print information
            print(f"Best mask shape: {best_mask.shape}")
            print(f"Best score: {best_score}")
            print(f"Total masks: {len(scores)}")
            
            # Create segmented image (background set to black)
            segmented_image = image.copy()
            segmented_image[~best_mask] = 0
            
            # Return results if requested
            if return_results:
                return {
                    "masks": masks,
                    "scores": scores,
                    "logits": logits,
                    "best_mask": best_mask,
                    "best_score": best_score,
                    "segmented_image": segmented_image,
                    "original_image": image
                }

        return 0

    except Exception as e:
        print(f"Error in main function: {e}")
        return -1


# if __name__ == "__main__":
#     # This allows running sam.py directly for testing
#     result = main()
#     if isinstance(result, dict):
#         print(f"Successfully processed image with SAM. Best score: {result['best_score']}")