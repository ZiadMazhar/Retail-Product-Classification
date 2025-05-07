import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Union

# Local imports
from sam import main as sam_main
from siamese import siameseModel, load_embedding_MobileNetV2
from cropper import cropper
from siamese import train as train_siamese


def load_siamese_model(
    model_path: str = "src/models/siamese_final_weights.h5",
) -> siameseModel:
    """
    Load the trained Siamese model.

    Args:
        model_path: Path to the saved model weights

    Returns:
        Loaded Siamese model
    """
    try:
        # Define image shape
        img_shape = (224, 224, 3)

        # Load embedding model
        embedding_model = load_embedding_MobileNetV2(img_shape)

        # Create Siamese model
        model = siameseModel(img_shape, embedding_model)

        # Build the model by calling it once with dummy data
        dummy_input = np.zeros((1, 2, *img_shape))
        _ = model(dummy_input)

        # Load weights
        model.load_weights(model_path)

        print(f"Successfully loaded model weights from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading Siamese model: {e}")
        return None


def preprocess_image(
    image: np.ndarray, target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Preprocess an image for the Siamese network.

    Args:
        image: Input image
        target_size: Target size for resizing

    Returns:
        Preprocessed image
    """
    # Resize image
    if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
        image = tf.image.resize(image, target_size).numpy()

    # Ensure image has 3 channels
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)

    # Normalize pixel values to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


def get_embedding(model: siameseModel, image: np.ndarray) -> np.ndarray:
    """
    Get the embedding of an image using the Siamese model.

    Args:
        model: Siamese model
        image: Preprocessed image

    Returns:
        Embedding vector
    """
    # Get the embedding model
    embedding_model = model.embedding_model

    # Get the embedding
    embedding = embedding_model.predict(image)

    return embedding


def comparator(zeta_Q: np.ndarray, zeta_P: np.ndarray) -> float:
    """
    Compare the two embeddings using the cosine similarity.

    Args:
        zeta_Q: The query embedding
        zeta_P: The positive embedding

    Returns:
        The cosine similarity
    """
    # Normalize the embeddings
    zeta_Q_norm = zeta_Q / np.linalg.norm(zeta_Q)
    zeta_P_norm = zeta_P / np.linalg.norm(zeta_P)

    # Calculate cosine similarity
    similarity = np.dot(zeta_Q_norm, zeta_P_norm)

    return similarity


def segment_and_crop_image(
    image_path: str, reference_shape: Tuple[int, int]
) -> Tuple[List[np.ndarray], List[Dict[str, int]]]:
    """
    Segment an image using SAM and crop all objects, only keeping crops similar in size to the reference image.

    Args:
        image_path: Path to the image
        reference_shape: (height, width) of the reference image

    Returns:
        Tuple of (list of cropped object images, list of bounding boxes)
    """
    # Run SAM to get segmentation
    results = sam_main(img_name=image_path, return_results=True)

    if results == -1 or not isinstance(results, dict):
        print("Error in segmentation")
        return [], []

    # Get all masks and original image
    bboxes = results["bboxes"]
    image = results["original_image"]
    original_height, original_width = image.shape[:2]

    # Store original image dimensions for later scaling
    image_dimensions = {"height": original_height, "width": original_width}

    ref_h, ref_w = reference_shape
    ref_area = ref_h * ref_w

    filtered_bboxes = []
    for b in bboxes:
        x, y, w, h = b
        x_min, y_min = int(x), int(y)
        x_max, y_max = int(x + w), int(y + h)
        filtered_bboxes.append([x_min, y_min, x_max, y_max])

    for idx, b in enumerate(filtered_bboxes):
        print(f"Filtered bbox {idx}: {b}, area: {(b[2]-b[0])*(b[3]-b[1])}")

    print(
        f"Reference area: {ref_area}, Filtered {len(filtered_bboxes)} out of {len(bboxes)} bboxes"
    )

    # Convert filtered_bboxes to a NumPy array with shape (N, 4) before cropping
    filtered_bboxes_np = np.array(filtered_bboxes, dtype=int).reshape(-1, 4)
    cropped_images = cropper(image, filtered_bboxes_np)

    # Convert the filtered bounding boxes to the format expected by the visualization code
    # [x_min, y_min, width, height]
    visualization_bboxes = []
    for bbox in bboxes:
        x_min, y_min, width, height = bbox
        visualization_bboxes.append(
            {
                "x": x_min,
                "y": y_min,
                "width": width,
                "height": height,
                "original_dimensions": image_dimensions,  # Store original image dimensions
            }
        )

    return cropped_images, visualization_bboxes


def find_matching_product(
    reference_img_path: str, multi_product_img_path: str
) -> Tuple[np.ndarray, float, Dict[str, int]]:
    """
    Find the product in the multi-product image that matches the reference product.

    Args:
        reference_img_path: Path to the reference product image
        multi_product_img_path: Path to the image with multiple products

    Returns:
        Tuple of (best_match_image, similarity_score, bounding_box)
    """
    # Load the Siamese model
    model = load_siamese_model()

    if model is None:
        print("Error loading model")
        return None, 0.0, None

    # Load the reference image
    image = tf.keras.preprocessing.image.load_img(
        reference_img_path, color_mode="rgb", target_size=(512, 512)
    )
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.array(image, dtype=np.uint8)

    # Preprocess reference image
    processed_reference = preprocess_image(image)
    processed_embedding = get_embedding(model, processed_reference)[0]
    ref_shape = image.shape[:2]  # (height, width)

    # Segment and crop all products in the multi-product image
    multi_product_crops, multi_product_bboxes = segment_and_crop_image(
        multi_product_img_path, ref_shape
    )

    if not multi_product_crops:
        print("Error: Could not segment any products in the multi-product image")
        return None, 0.0, None

    # Find the best match
    best_match = None
    best_similarity = -1.0
    best_bbox = None

    for i, crop in enumerate(multi_product_crops):
        print(
            f"Crop {i} shape: {crop.shape if isinstance(crop, np.ndarray) else 'Not ndarray'}"
        )
        # Preprocess crop
        processed_crop = preprocess_image(crop)

        # Get embedding
        crop_embedding = get_embedding(model, processed_crop)[0]

        # Compare embeddings
        similarity = comparator(processed_embedding, crop_embedding)

        print(f"Similarity for crop {i}: {similarity}")

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = crop
            best_bbox = (
                multi_product_bboxes[i] if i < len(multi_product_bboxes) else None
            )

    return best_match, best_similarity, best_bbox


def test() -> int:
    """
    Test the product matching functionality.

    Args:
        None

    Returns:
        int, 0 for success, 1 for failure
    """
    try:
        # Get paths from user
        reference_img_path = r"C:\Users\Ziad Mazhar\Documents\GitHub\Retail-Product-Classification\dbs\comparator_db\raw\P\1234599_kelloggs-coco-pops-375g-box.png.jpg"
        # multi_product_img_path = r"C:\Users\Ziad Mazhar\Downloads\19976638-7595095-image-a-31_1571634857477.jpg"
        multi_product_img_path = r"C:\Users\Ziad Mazhar\Downloads\640.webp"
        # multi_product_img_path = r"C:\Users\Ziad Mazhar\Downloads\rice-krispies-and-coco-pops-cereal-stacked-own-the-shelf-in-a-uk-supermarket-ke44ek.jpg"
        # Strip quotes if present
        reference_img_path = reference_img_path.strip("\"'")
        multi_product_img_path = multi_product_img_path.strip("\"'")

        # Check if files exist
        if not os.path.exists(reference_img_path):
            print(f"Reference image not found: {reference_img_path}")
            return 1

        if not os.path.exists(multi_product_img_path):
            print(f"Multi-product image not found: {multi_product_img_path}")
            return 1

        print("Processing images...")

        # Find matching product
        matching_product, similarity, bbox = find_matching_product(
            reference_img_path, multi_product_img_path
        )

        if matching_product is None:
            print("No matching product found")
            return 1

        # Display results
        plt.figure(figsize=(15, 5))

        # Load and display reference image
        reference_img = plt.imread(reference_img_path)
        plt.subplot(1, 3, 1)
        plt.imshow(reference_img)
        plt.title("Reference Product")
        plt.axis("off")

        # Display matching product
        plt.subplot(1, 3, 2)
        plt.imshow(matching_product.astype(np.uint8))
        plt.title(f"Matching Product\nSimilarity: {similarity:.4f}")
        plt.axis("off")

        # Display original multi-product image with bounding box
        multi_product_img = plt.imread(multi_product_img_path)
        plt.subplot(1, 3, 3)
        plt.imshow(multi_product_img)

        # Draw the bounding box if available
        if bbox is not None:
            from matplotlib.patches import Rectangle

            # Get bounding box coordinates
            x = bbox["x"]
            y = bbox["y"]
            width = bbox["width"]
            height = bbox["height"]

            # Get original image dimensions if available
            if "original_dimensions" in bbox:
                orig_dims = bbox["original_dimensions"]
                img_height, img_width = multi_product_img.shape[:2]

                # Scale bounding box if the displayed image dimensions differ from original
                if orig_dims["height"] != img_height or orig_dims["width"] != img_width:
                    x_scale = img_width / orig_dims["width"]
                    y_scale = img_height / orig_dims["height"]

                    x = int(x * x_scale)
                    y = int(y * y_scale)
                    width = int(width * x_scale)
                    height = int(height * y_scale)

                    print(f"Scaled bbox: x={x}, y={y}, width={width}, height={height}")

            # Create a rectangle patch with a thicker line and more visible color
            rect = Rectangle(
                (x, y), width, height, linewidth=3, edgecolor="lime", facecolor="none"
            )
            plt.gca().add_patch(rect)

            # Add a text label near the bounding box
            plt.text(
                x,
                y - 10,
                f"Match: {similarity:.2f}",
                color="white",
                fontsize=12,
                bbox=dict(facecolor="red", alpha=0.7),
            )

        plt.title("Multi-Product Image\nMatch Highlighted")
        plt.axis("off")

        plt.tight_layout()
        # plt.savefig("matching_result.png")  # Save the visualization
        plt.show()

        print(f"Matching product found with similarity: {similarity:.4f}")
        print(
            f"{'Match confirmed' if similarity > 0.7 else 'Possible match, but similarity is low'}"
        )
        if bbox:
            print(
                f"Matching product location: x={bbox['x']}, y={bbox['y']}, width={bbox['width']}, height={bbox['height']}"
            )

        return 0

    except Exception as e:
        print(f"Error in test function: {e}")
        import traceback

        traceback.print_exc()  # Print full traceback for debugging
        return 1


def main():
    """
    Main function to run the application.
    """
    print("Retail Product Classification")
    print("1. Find matching product")
    print("2. Train Siamese model")
    print("3. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        test()
    elif choice == "2":
        epochs = int(input("Enter number of training epochs (recommended: 10-30): "))
        verbose = input("Enable verbose output? (y/n): ").lower() == "y"

        print(f"Starting Siamese model training with {epochs} epochs...")
        status = train_siamese(total_epochs=epochs, verbose=verbose)

        if status == 0:
            print("Training completed successfully!")
        else:
            print("Training failed with status code:", status)
    elif choice == "3":
        print("Exiting...")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
