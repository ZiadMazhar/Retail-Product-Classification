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
    model_type: str = "vit_b",
    checkpoint_path: str = "src/models/sam_vit_b.pth",
    return_results: bool = False,
) -> Union[int, Dict[str, Any]]:
    """
    Main function for the SAM model that either generates or loads the masks for a given image
    
    Args:
        img_name: the name of the image to load
        model_type: SAM model type (vit_b, vit_l, vit_h)
        checkpoint_path: path to the model checkpoint
        return_results: whether to return the results dictionary instead of status code
        
    Returns:
        If return_results is False: 0 if successful, -1 if failed
        If return_results is True: Dictionary with bboxes, orginal image
    """
    try:
        # Check if image file exists
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image file not found: {img_name}")
        sam_helper_instance = SamHelper()
        # Initialize the SAM model
        sam_helper_instance.init_model(model_type=model_type, model_path=checkpoint_path)
        
        # Load the image
        image = tf.keras.preprocessing.image.load_img(
            img_name, color_mode="rgb", target_size=(512, 512)
        )
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.array(image, dtype=np.uint8)

        # Use automatic mask generation
        bboxes = sam_helper_instance.generate_masks(image)
        print(f"Generated {len(bboxes)} bounding boxes automatically")
                
        if return_results:
            return {
                "bboxes": bboxes,
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