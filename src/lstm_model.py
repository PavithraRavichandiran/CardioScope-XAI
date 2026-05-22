"""
LSTM temporal model for CardioScope-XAI.

Architecture:
    Input  (batch, 12, 4)
      → LSTM(64, return_sequences=True)
      → Dropout(0.3)
      → LSTM(32)
      → Dropout(0.2)
      → Dense(16, relu)   ← temporal latent embedding extracted from here
      → Dense(1, sigmoid) ← standalone LSTM prediction

The 16-dim Dense layer output is the temporal embedding used in fusion.
"""

import numpy as np
import joblib
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import mlflow
import mlflow.keras

MODEL_PATH     = Path("models/lstm_model.keras")
ENCODER_PATH   = Path("models/lstm_encoder.keras")
MLFLOW_TRACKING = "mlflow"
EXPERIMENT_NAME = "CardioScope-XAI / LSTM"
RANDOM_STATE    = 42

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def build_lstm_model(input_shape: tuple = (12, 4)) -> tuple[Model, Model]:
    """
    Build the full LSTM classifier and a separate encoder model
    that outputs the 16-dim temporal latent embedding.

    Returns:
        full_model   : LSTM → prediction (for training)
        encoder_model: LSTM → 16-dim embedding (for fusion)
    """
    inputs = Input(shape=input_shape, name="temporal_input")

    x = LSTM(64, return_sequences=True, name="lstm_1")(inputs)
    x = Dropout(0.3, name="dropout_1")(x)
    x = LSTM(32, return_sequences=False, name="lstm_2")(x)
    x = Dropout(0.2, name="dropout_2")(x)

    # Temporal embedding layer — output of this is used in fusion
    embedding = Dense(16, activation="relu", name="temporal_embedding")(x)

    output = Dense(1, activation="sigmoid", name="risk_output")(embedding)

    full_model    = Model(inputs, output,    name="lstm_classifier")
    encoder_model = Model(inputs, embedding, name="lstm_encoder")

    return full_model, encoder_model


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    log_mlflow: bool = True,
) -> tuple[Model, Model, dict]:
    """
    Train the LSTM classifier.

    Returns:
        full_model    : trained LSTM classifier
        encoder_model : trained encoder (shares weights with full_model)
        history_dict  : training history metrics
    """
    full_model, encoder_model = build_lstm_model(input_shape=X_train.shape[1:])

    full_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        EarlyStopping(monitor="val_recall", patience=15, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5, verbose=0),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_recall", save_best_only=True,
                        mode="max", verbose=0),
    ]

    # Class weights to boost recall on disease class
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    class_weight = {0: 1.0, 1: neg / pos}

    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="lstm_baseline"):
        history = full_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        # Evaluate on validation set
        val_loss, val_acc, val_recall, val_auc = full_model.evaluate(X_val, y_val, verbose=0)
        metrics = {
            "val_accuracy": round(float(val_acc),    4),
            "val_recall":   round(float(val_recall), 4),
            "val_auc":      round(float(val_auc),    4),
            "val_loss":     round(float(val_loss),   4),
        }

        if log_mlflow:
            mlflow.log_params({
                "epochs": epochs, "batch_size": batch_size,
                "learning_rate": learning_rate,
                "architecture": "LSTM(64)->LSTM(32)->Dense(16)->Dense(1)",
            })
            mlflow.log_metrics(metrics)

        # Save encoder (shares trained weights via shared layers)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        encoder_model.save(ENCODER_PATH)
        print(f"[lstm_model] Full model saved  → {MODEL_PATH}")
        print(f"[lstm_model] Encoder saved     → {ENCODER_PATH}")

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    return full_model, encoder_model, history_dict


def load_model() -> Model:
    """Load the saved full LSTM classifier."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"LSTM model not found at {MODEL_PATH}. Run training first.")
    return tf.keras.models.load_model(MODEL_PATH)


def load_encoder() -> Model:
    """Load the saved LSTM encoder (for fusion)."""
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"LSTM encoder not found at {ENCODER_PATH}. Run training first.")
    return tf.keras.models.load_model(ENCODER_PATH)


def extract_embeddings(encoder: Model, sequences: np.ndarray) -> np.ndarray:
    """
    Extract 16-dim temporal embeddings from all patient sequences.

    Args:
        encoder   : trained lstm_encoder model
        sequences : (n_patients, n_timesteps, n_features)

    Returns:
        embeddings: (n_patients, 16)
    """
    return encoder.predict(sequences, verbose=0)


def predict_proba_single(full_model: Model, sequence: np.ndarray) -> float:
    """Return disease probability for one patient's temporal sequence."""
    seq = sequence[np.newaxis, ...]  # add batch dim
    return float(full_model.predict(seq, verbose=0)[0, 0])
