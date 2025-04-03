import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense,
    Input,
    Lambda,
    GlobalAveragePooling2D,
    Concatenate,
)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

### locak import ###
from data.siamese_data_pipeline import data_pipeline

### type definitions  ###
# none


### load up the env var ###
class siameseModel:
    def __init__(self, img_shape: tuple[int, ...], embedding_model: k.Model):
        """
        Initialize the Siamese network model.

        Args:
            img_shape: Shape of input images (height, width, channels)
            embedding_model: Pre-trained model for feature extraction
        """
        super(siameseModel, self).__init__()
        self.image_shape = img_shape
        self.embedding_model = embedding_model
        self.distance_layer = Lambda(
            lambda embeddings: tf.math.abs(embeddings[0] - embeddings[1], axis=1),
            output_shape=lambda shapes: shapes[0],
        )
        self.classifier = k.Sequential(
            [
                Dense(128, activation="relu"),
                Dense(64, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )
        self._build_model()

    def _build_model(self):
        """Build the complete Siamese model architecture."""
        # Input for image pairs
        input_a = Input(shape=self.img_shape)
        input_b = Input(shape=self.img_shape)

        # Get embeddings for both inputs
        embedding_a = self.embedding_model(input_a)
        embedding_b = self.embedding_model(input_b)

        # concating the two embeddings
        abs_diff = Lambda(lambda x: k.abs(x[0] - x[1]))([embedding_a, embedding_b])
        merged = Concatenate(axis=-1)([embedding_a, embedding_b, abs_diff])

        # Pass through classifier to get similarity score
        outputs = self.classifier(merged)

        # Create the trainable model
        self.siamese_net = Model(inputs=[input_a, input_b], outputs=outputs)

    def call(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the Siamese network.

        Args:
            X: Input tensor of shape (2, batch_size, height, width, channels)
                X[0] is the first set of images
                X[1] is the second set of images

        Returns:
            Similarity scores between pairs of images
        """
        image_a, image_b = X
        return self.siamese_net([image_a, image_b])

    def train_step(self, data: tuple[np.ndarray]) -> dict[str, float]:
        """
        custom training step for the Siamese network.
          args:
               data: tuple[np.ndarray] :
               data[0] pair of images
               data[1] target  0 or 1
             returns:
             dictoionary of metrics
        """
        # Unpack the data
        X, y = data

        # GradientTape records operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(X, training=True)

            # Compute the loss
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current values
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: tuple[np.ndarray]) -> dict[str, float]:
        """
        custom test step for the Siamese network.
        args:
        data: tuple[np.ndarray] :
        data[0] pair of images
        data[1] target  0 or 1
        returns:
        dictoionary of metrics
        """
        # Unpack the data
        X, y = data
        # Compute predictions
        y_pred = self(X, training=False)
        # calculate the loss
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current values
        return {m.name: m.result() for m in self.metrics}
        ...

    def load(self, path: str) -> None:
        """
        Load the model weights from a file.
        Args:
        path: str, path to the file containing the model weights.
        """
        try:
            self.load_weights(path)
            print(f"Model weights loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")

    def freeze_embedding_model(self) -> None:
        """Freeze the weights of the embedding model."""
        for layer in self.embedding_model.layers:
            layer.trainable = False
        print("Embedding model frozen.")

    def unfreeze_embedding_model(self) -> None:
        """Unfreeze the weights of the embedding model."""
        for layer in self.embedding_model.layers:
            layer.trainable = True
        print("Embedding model unfrozen.")


def load_embedding_MobileNetV2(img_shape: tuple[int, ...]) -> k.Model:
    """
    loads the MobileNetV2 model to be used as the embedding model.
    args:
        img_shape: tuple[int], the shape of the images.
    returns:
        model: k.Model, the MobileNetV2 model.
    """
    # Load the MobileNetV2 model woithout the top layer
    base_model = MobileNetV2(
        input_shape=img_shape, include_top=False, weights="imagenet"
    )
    # Apply Global Average Pooling to convert feature maps into a compact embedding
    x = GlobalAveragePooling2D()(base_model.output)
    # Create the Embedding model
    model = Model(inputs=base_model.input, outputs=x)
    # Print model summary
    print(f"MobileNetV2 embedding model loaded with output shape: {model.output_shape}")

    return model


# tensor board


def plot_training_history(history1, history2=None, save_path=None):
    """
    Plot training and validation metrics.

    Args:
        history1: History from first training phase
        history2: History from second training phase (optional)
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))

    # Plot training & validation accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history1.history["accuracy"], label="Phase 1 Training")
    plt.plot(history1.history["val_accuracy"], label="Phase 1 Validation")

    if history2:
        # Adjust x-axis for phase 2
        x_offset = len(history1.history["accuracy"])
        x_phase2 = [x + x_offset for x in range(len(history2.history["accuracy"]))]

        plt.plot(x_phase2, history2.history["accuracy"], label="Phase 2 Training")
        plt.plot(x_phase2, history2.history["val_accuracy"], label="Phase 2 Validation")

    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="lower right")
    plt.grid(True)

    # Plot training & validation loss
    plt.subplot(2, 1, 2)
    plt.plot(history1.history["loss"], label="Phase 1 Training")
    plt.plot(history1.history["val_loss"], label="Phase 1 Validation")

    if history2:
        # Adjust x-axis for phase 2
        plt.plot(x_phase2, history2.history["loss"], label="Phase 2 Training")
        plt.plot(x_phase2, history2.history["val_loss"], label="Phase 2 Validation")

    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def create_lr_schedule(
    initial_lr: float, min_lr: float, epochs: int
) -> k.optimizers.schedules.LearningRateSchedule:
    """
    Create a learning rate schedule for annealing.

    Args:
        initial_lr: Initial learning rate
        min_lr: Minimum learning rate
        epochs: Number of epochs for annealing

    Returns:
        Learning rate schedule
    """
    # Calculate decay rate to reach min_lr by the end of training
    decay_rate = (min_lr / initial_lr) ** (1 / epochs)

    # Create an exponential decay schedule
    lr_schedule = k.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=1,  # decay after each epoch
        decay_rate=decay_rate,
        staircase=True,
    )

    return lr_schedule


def train(total_epochs: int = 10, verbose: bool = False) -> int:
    """
    Train the Siamese network using a two-phase approach with learning rate annealing.

    Args:
        total_epochs: Total number of training epochs
        verbose: Whether to print progress information

    Returns:
        status_code: Integer indicating training success/failure
    """
    # init the progress bar
    if verbose:
        print(f"{'running the data pipeline':=^40}")
    # calculate the number of epochs
    first_phase_epochs = int(total_epochs * 0.3)
    second_phase_epochs = total_epochs - first_phase_epochs
    if verbose:
        print(f"first phase epochs: {first_phase_epochs}")
        print(f"second phase epochs: {second_phase_epochs}")
    # define image shape
    img_shape = (224, 224, 3)
    batch_size = 32

    (X_train, y_train), (X_test, y_test) = data_pipeline(
        root_in_path="dbs/comparator_db/raw",
        X_shape=(img_shape[0], img_shape[1]),
        num_augment=4,
        train_ratio=0.8,
        verbose=verbose,
    )
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = (
        train_dataset.shuffle(buffer_size=1024)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # create model
    embedding_model = load_embedding_MobileNetV2(img_shape)
    model = siameseModel(img_shape, embedding_model)
    # compile the model
    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=1e-3),
        loss=k.losses.BinaryCrossentropy(),
        metrics=[k.metrics.BinaryAccuracy()],
    )
    # train the model as per the two phase approach
    try:
        # first phase training
        if verbose:
            print(f"{'first phase training':=^40}")
        # freeze the embedding model
        model.freeze_embedding_model()
        # train for the first phase
        history1 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=first_phase_epochs,
            verbose=1 if verbose else 0,
        )
        model.save("models/siamese_model_first_phase.h5")
        # second phase training
        if verbose:
            print(f"{'second phase training':=^40}")
        # unfreeze the embedding model
        model.unfreeze_embedding_model()
        # create the learning rate schedule
        lr_schedule = create_lr_schedule(
            initial_lr=5e-4,  # Lower initial rate for fine-tuning
            min_lr=1e-6,
            epochs=second_phase_epochs,
        )
        # compile the model with the new learning rate
        model.compile(
            optimizer=k.optimizers.Adam(learning_rate=lr_schedule),
            loss=k.losses.BinaryCrossentropy(),
            metrics=[k.metrics.BinaryAccuracy()],
        )
        # train for the second phase
        history2 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=second_phase_epochs,
            verbose=1 if verbose else 0,
            callbacks=[
                k.callbacks.ModelCheckpoint(
                    "models/siamese_model_second_phase",
                    save_best_only=True,
                    monitor="val_accuracy",
                ),
                k.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True, monitor="val_accuracy"
                ),
            ],
        )
        # save the model
        model.save("models/siamese_fina.h5")
        # Plot training history
        if verbose:
            plot_training_history(
                history1, history2, save_path="models/training_history.png"
            )

        return 0  # Success
    except Exception as e:
        print(f"Error during tarining: {e}")
        return 1
