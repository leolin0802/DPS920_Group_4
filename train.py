"""
train.py
DPS920 Final Project - Self-Driving Car Simulation
---------------------------------------------------------------------------
PURPOSE:
    Build the NVIDIA self-driving CNN (Figure 7 in the project spec),
    train it using the batch generator from dataPreprocessing.py,
    plot the loss curves, and save the trained model as 'model.h5'.

USAGE:
    python train.py

    Training takes ~10-30 min depending on dataset size and whether you
    have a GPU available.  Watch the loss curves – both train and val loss
    should decrease smoothly.  If val loss starts rising while train loss
    keeps falling, the model is overfitting (see notes at the bottom).

OUTPUT:
    model.h5       – the saved Keras model, used by TestSimulation.py
    training_plot.png  – loss curve saved to disk
---------------------------------------------------------------------------
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress TensorFlow C++ warnings

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, Flatten, Dense, Dropout, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Local modules
from dataPreprocessing import prepareData, batchGenerator


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS  (adjust these if training doesn't converge)
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH        = 'data'    # folder containing driving_log.csv + IMG/
BATCH_SIZE       = 100       # samples per gradient update
EPOCHS           = 10        # number of full passes through the generator
LEARNING_RATE    = 1e-3      # Adam learning rate
STEPS_PER_EPOCH  = 300       # generator steps per epoch (= samples seen / batch)
VALIDATION_STEPS = 200       # generator steps for validation


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITION  –  NVIDIA Architecture (Figure 7)
# ══════════════════════════════════════════════════════════════════════════════

def buildModel():
    """
    Construct the NVIDIA end-to-end self-driving CNN as described in Figure 7
    of the project spec.

    Architecture summary
    --------------------
    Input         : (66, 200, 3)  – YUV image after preprocessing
    Lambda        : normalisation layer (pixels already in [0,1] from preprocessing,
                    this lambda re-centres to [-1, 1] which helps gradient flow)
    Conv1         : 24 filters, 5×5, stride 2, ELU
    Conv2         : 36 filters, 5×5, stride 2, ELU
    Conv3         : 48 filters, 5×5, stride 2, ELU
    Conv4         : 64 filters, 3×3, ELU
    Conv5         : 64 filters, 3×3, ELU
    Flatten
    Dense(100)    : ELU
    Dense(50)     : ELU
    Dense(10)     : ELU
    Dense(1)      : linear output → steering angle

    Notes
    -----
    - ELU (Exponential Linear Unit) is used instead of ReLU because it allows
      small negative outputs, which is important when predicting negative
      steering angles.
    - Dropout(0.5) is added after Flatten to reduce overfitting.
    - The output layer has NO activation (linear) because steering is a
      continuous regression target, not a classification.
    """

    model = Sequential([

        # Normalisation layer: maps [0,1] → [-1, 1]
        Lambda(lambda x: x / 0.5 - 1.0, input_shape=(66, 200, 3)),

        # ── Convolutional feature extractor ──────────────────────────────
        # Three 5×5 conv layers with stride=2 (downsampling built in)
        Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),

        # Two 3×3 conv layers (no striding – fine-grained feature extraction)
        Conv2D(64, (3, 3), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),

        # ── Fully-connected classifier / regressor ────────────────────────
        Flatten(),
        Dropout(0.5),          # regularisation – drop 50 % of neurons randomly

        Dense(100, activation='elu'),
        Dense(50,  activation='elu'),
        Dense(10,  activation='elu'),

        # Single output neuron: predicted steering angle (linear)
        Dense(1),
    ])

    # Mean Squared Error loss is standard for regression tasks.
    # Adam optimizer adapts the learning rate automatically.
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=LEARNING_RATE)
    )

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def trainModel():
    """
    Full training pipeline:
        1. Prepare data splits.
        2. Build the model.
        3. Train with generators.
        4. Plot and save loss curves.
        5. Save model.h5.
    """

    # ── Step 1: Data ──────────────────────────────────────────────────────
    print('\n[train] Loading and balancing data...')
    xTrain, xVal, yTrain, yVal = prepareData(DATA_PATH, display=False)

    trainGen = batchGenerator(xTrain, yTrain, BATCH_SIZE, isTraining=True)
    valGen   = batchGenerator(xVal,   yVal,   BATCH_SIZE, isTraining=False)

    # ── Step 2: Model ─────────────────────────────────────────────────────
    print('\n[train] Building NVIDIA CNN...')
    model = buildModel()
    model.summary()

    # ModelCheckpoint: automatically save the best model (lowest val_loss)
    # so even if training crashes mid-way, you keep the best weights.
    checkpoint = ModelCheckpoint(
        'model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # ── Step 3: Training ──────────────────────────────────────────────────
    print('\n[train] Starting training...')
    history = model.fit(
        trainGen,
        steps_per_epoch  = STEPS_PER_EPOCH,
        epochs           = EPOCHS,
        validation_data  = valGen,
        validation_steps = VALIDATION_STEPS,
        callbacks        = [checkpoint],
        verbose          = 1
    )

    # ── Step 4: Loss curves ───────────────────────────────────────────────
    print('\n[train] Plotting loss curves...')
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'],     label='Training Loss',   color='steelblue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Model Training Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_plot.png')
    plt.show()
    print('[train] Plot saved to training_plot.png')

    # ── Step 5: Save model ────────────────────────────────────────────────
    # ModelCheckpoint already saves the BEST epoch to model.h5 above.
    # We also do an explicit final save here as a backup.
    model.save('model.h5')
    print('\n[train] Model saved to model.h5  ✓')
    print('[train] Training complete!')


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    trainModel()


# ══════════════════════════════════════════════════════════════════════════════
#  TROUBLESHOOTING NOTES
# ══════════════════════════════════════════════════════════════════════════════
#
#  Problem: val_loss is much higher than loss (overfitting)
#  Fix: reduce EPOCHS, increase Dropout rate, collect more data
#
#  Problem: both losses are high / not decreasing (underfitting)
#  Fix: increase EPOCHS, lower LEARNING_RATE (e.g. 1e-4), collect more data
#
#  Problem: car drives straight and ignores curves
#  Fix: re-balance the dataset (lower the threshold in balanceData()),
#       check that driving data includes enough sharp turns
#
#  Problem: "out of memory" error during training
#  Fix: reduce BATCH_SIZE to 50 or lower
