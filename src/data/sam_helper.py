import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt  # Fix: import pyplot directly, not matplotlib as plt
import os

### locak import ###
# none

### type definitions  ###
# none

### load up the env var ###
# none


class SamHelper:
    def init_model(
        self,
        model_type="vit_b",
        model_path="src/models/sam_vit_b.pth",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the model

        Args:
        model_type : str : the name of the model to be used
        device : str : the device to be used
        model_path : str : the path to the model file
        return : SamPredictor : the predictor object
        """
        self.model_type = model_type
        # Normalize path separators for the current OS
        self.checkpoint_path = os.path.normpath(model_path)
        self.device = device
        self.sam = None
        self.predictor = None
        self._load_model()

    def _load_model(self):
        """
        Load the same model and intialize the predictor
        """
        try:
            # Check if file exists and is accessible
            if not os.path.exists(self.checkpoint_path):
                print(f"Model file not found: {self.checkpoint_path}")
                return
                
            # Try to open the file to check permissions
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    pass
            except PermissionError:
                print(f"Permission denied when accessing model file: {self.checkpoint_path}")
                print("Try running the application as administrator or check file permissions")
                return
                
            # Load the model
            self.sam = sam_model_registry[self.model_type](
                checkpoint=self.checkpoint_path
            )
            self.sam = self.sam.to(self.device)
            self.predictor = SamPredictor(self.sam)
            print(f"Model {self.model_type} loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading the SAM model: {e}")

    def set_image(self, image):
        """
        Set the image to be processed by the predictor

        Args:
        image : numpy.ndarray : the RGB image to be processed
        """
        self.predictor.set_image(image)
        return self

    def predict_masks_from_points(
        self, points, point_labels=None, multimask_output=True
    ):
        """
        Predict the masks from the points

        Args:
        points : list : the list of points to be predicted
        point_labels : list : the list of labels for the points ( 1 for foreground, 0 for background)
        multimask_output : bool : whether to return the mask as a single mask or multiple masks
        returns
        masks : numpy.ndarray : the predicted masks
        scores : numpy.ndarray : the confidence of the predicted masks
        logits : numpy.ndarray : the raw logits  of the model

        """
        try:
            if point_labels is None:
                # Convert points to numpy array if it's a list
                if isinstance(points, list):
                    points = np.array(points)
                point_labels = np.ones(len(points), dtype=np.int32)
            
            # Ensure points is a numpy array
            if isinstance(points, list):
                points = np.array(points)
                
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                multimask_output=multimask_output,
            )
            return masks, scores, logits
        except Exception as e:
            print(f"Error predicting masks: {e}")
            # Return empty arrays instead of None to avoid unpacking errors
            return np.array([]), np.array([]), np.array([])

    def predict_masks_from_box(self, box, multimask_output=True):
        """
        Predict the masks from the bounding boxes

        Args:
        box (numpy.ndarray): Bounding box in xyxy format
        multimask_output : bool : whether to return the mask as a single mask or multiple masks
        returns
        masks : numpy.ndarray : the predicted masks
        scores : numpy.ndarray : the confidence of the predicted masks
        logits : numpy.ndarray : the raw logits  of the model
        """
        try:
            # Fix typo: self.predecitor -> self.predictor
            masks, scores, logits = self.predictor.predict(
                box=box, multimask_output=multimask_output
            )
            return masks, scores, logits
        except Exception as e:
            print(f"Error predicting masks: {e}")
            # Return empty arrays instead of None
            return np.array([]), np.array([]), np.array([])

    def predict_masks_from_prompt(
        self, points, point_labels=None, box=None, multimask_output=True
    ):
        """
        Predict the masks from the points

        Args:
        points : list : the list of points to be predicted
        point_labels : list : the list of labels for the points ( 1 for foreground, 0 for background)
        box (numpy.ndarray): Bounding box in xyxy format
        multimask_output : bool : whether to return the mask as a single mask or multiple masks
        returns
        masks : numpy.ndarray : the predicted masks
        scores : numpy.ndarray : the confidence of the predicted masks
        logits : numpy.ndarray : the raw logits  of the model
        """
        try:
            # Convert points to numpy array if it's a list
            if isinstance(points, list):
                points = np.array(points)
                
            masks, scores, logits = self.predictor.predict(
                box=box,
                point_coords=points,
                point_labels=point_labels,
                multimask_output=multimask_output,
            )
            return masks, scores, logits
        except Exception as e:
            print(f"Error predicting masks: {e}")
            # Return empty arrays instead of None
            return np.array([]), np.array([]), np.array([])

    @staticmethod
    def visualize_masks(image, masks, scores=None, alpha=0.5):
        """
        Visualize the masks on the image

        Args:
        image : numpy.ndarray : the image to be visualized
        masks : numpy.ndarray : the masks to be visualized
        scores : numpy.ndarray : the confidence of the masks
        alpha : float : the transparency of the masks
        returns
        fig : matplotlib.figure.Figure : Figure with the visualized masks
        """
        try:
            # Check if masks is empty
            if masks.size == 0:
                print("No masks to visualize")
                return None
                
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(image)
            
            # Sort masks by scores if scores are provided
            if scores is not None and scores.size > 0:
                idx = np.argsort(scores)[::-1]
                masks = masks[idx]
                scores = scores[idx]

            colors = np.random.rand(len(masks), 3)
            for i, (mask, color) in enumerate(zip(masks, colors)):
                colored_mask = np.zeros_like(image)
                colored_mask[mask] = color
                plt.imshow(colored_mask, alpha=alpha * mask)

                if scores is not None:
                    plt.text(
                        10,
                        20 + 20 * i,
                        f"Score: {scores[i]:.3f}",
                        fontsize=12,
                        color=color,
                        backgroundcolor="black",
                    )
            plt.axis("off")
            return fig
        except Exception as e:
            print(f"Error visualizing masks: {e}")
            return None

    @staticmethod
    def extract_best_mask(masks, scores):
        """
        Extract the best mask from the masks

        Args:
        masks : numpy.ndarray : the masks to be extracted
        scores : numpy.ndarray : the confidence of the masks
        returns
        best_mask : numpy.ndarray : the best mask
        best_score : float : the confidence of the best mask
        """
        try:
            # Check if masks or scores are empty
            if masks.size == 0 or scores.size == 0:
                print("No masks or scores available to extract best mask")
                return None, 0.0
                
            idx = np.argmax(scores)
            return masks[idx], scores[idx]
        except Exception as e:
            print(f"Error extracting best mask: {e}")
