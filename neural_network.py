"""
Neural Network module — Activation Function Comparison
Uses TensorFlow / Keras to build, train and evaluate models
with Sigmoid, Tanh, ReLU, and Leaky ReLU activation functions.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF info messages

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


def build_model(input_dim, activation='relu'):
    """
    Build a 3-hidden-layer neural network.

    Architecture:
        Input(input_dim) → 32 → 16 → 8 → 1(sigmoid)

    For Leaky ReLU we use a separate Keras layer because
    it cannot be passed as a simple string.
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    if activation == 'leaky_relu':
        # Leaky ReLU as explicit layers
        model.add(Dense(32))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        model.add(Dense(16))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        model.add(Dense(8))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    else:
        model.add(Dense(32, activation=activation))
        model.add(Dense(16, activation=activation))
        model.add(Dense(8, activation=activation))

    # Output layer — sigmoid for binary classification
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_nn_with_activation(X_train, y_train, X_test, y_test,
                              activation='relu', epochs=100):
    """
    Train a neural network with the given activation function
    and return test accuracy.
    """
    # Fix seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    model = build_model(X_train.shape[1], activation)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return round(accuracy, 4)
