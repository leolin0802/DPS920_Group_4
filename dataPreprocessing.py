"""
dataPreprocessing.py
DPS920 Final Project - Self-Driving Car Simulation
---------------------------------------------------------------------------
PURPOSE:
    1. Load & balance data (via dataCollection.py helpers).
    2. Split into train / validation sets.
    3. Define all augmentation functions (flip, brightness, zoom, pan).
    4. Define the preprocessing pipeline (crop → YUV → blur → resize → normalise).
    5. Provide a batch generator that applies augmentation on-the-fly during
       training (augmentation is ONLY applied to training batches, NOT validation).

USAGE:
    This script is imported by train.py.
    You can also run it standalone to verify images look correct.

        python dataPreprocessing.py

EXPECTED INPUTS:
    data/driving_log.csv
    data/IMG/
---------------------------------------------------------------------------
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from sklearn.model_selection import train_test_split

# Local module (must be in the same folder)
from dataCollection import loadData, balanceData


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – AUGMENTATION FUNCTIONS
#  Applied RANDOMLY to training images only (not validation).
#  Each function takes (image, steering) and returns (augmented_image, new_steering).
# ══════════════════════════════════════════════════════════════════════════════

def augmentFlip(image, steering):
    """
    Horizontally flip the image with 50 % probability.
    When flipped the steering direction is reversed (multiply by -1).
    This doubles effective dataset size and balances left/right turns.
    """
    if random.random() > 0.5:
        image    = cv2.flip(image, 1)   # 1 = horizontal flip
        steering = -steering
    return image, steering


def augmentPan(image, steering):
    """
    Randomly shift (pan) the image horizontally and vertically by up to
    10 % of image dimensions.  A horizontal pan simulates the car being
    off-centre, so we nudge the steering angle slightly.
    """
    h, w = image.shape[:2]
    tx   = w * (random.random() - 0.5) * 0.2   # ±10 % horizontal
    ty   = h * (random.random() - 0.5) * 0.2   # ±10 % vertical
    M    = np.float32([[1, 0, tx], [0, 1, ty]])
    image    = cv2.warpAffine(image, M, (w, h))
    steering = steering + tx / w * 0.3           # small proportional correction
    return image, steering


def augmentBrightness(image, steering):
    """
    Randomly adjust brightness by converting to HSV, scaling the V channel,
    then converting back to RGB.  This teaches the model to generalise across
    different lighting conditions (day / shadows / tunnel).
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    factor = 0.5 + random.random()              # scale factor in [0.5, 1.5]
    image[:, :, 2] = np.clip(image[:, :, 2] * factor, 0, 255)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return image, steering


def augmentZoom(image, steering):
    """
    Randomly zoom into the centre of the image by up to 30 %.
    The zoomed region is resized back to the original dimensions so
    downstream preprocessing receives a consistent shape.
    """
    h, w   = image.shape[:2]
    factor = 1 + random.random() * 0.3          # zoom factor in [1.0, 1.3]
    new_h  = int(h / factor)
    new_w  = int(w / factor)
    y1     = (h - new_h) // 2
    x1     = (w - new_w) // 2
    image  = image[y1:y1 + new_h, x1:x1 + new_w]
    image  = cv2.resize(image, (w, h))
    return image, steering


def randomAugment(image, steering):
    """
    Apply each augmentation function randomly and independently.
    Called ONLY on training samples inside the batch generator.
    """
    image, steering = augmentFlip(image, steering)
    image, steering = augmentPan(image, steering)
    image, steering = augmentBrightness(image, steering)
    image, steering = augmentZoom(image, steering)
    return image, steering


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – PREPROCESSING PIPELINE
#  These exact steps are used BOTH during training and at inference time
#  (TestSimulation.py already contains the same pipeline – keep them in sync!)
# ══════════════════════════════════════════════════════════════════════════════

def preProcessing(img):
    """
    Apply the mandatory preprocessing steps from the project spec (Section 5):

    1. Crop  – remove sky + hood (rows 60–135) to focus on the road.
    2. YUV   – convert colour space as required by the NVIDIA model.
    3. Blur  – Gaussian blur to reduce noise.
    4. Resize – to 200×66 as expected by the NVIDIA architecture.
    5. Normalise – scale pixel values to [0, 1].

    Parameters
    ----------
    img : np.ndarray  (H × W × 3, RGB uint8)

    Returns
    -------
    img : np.ndarray  (66 × 200 × 3, float32  in [0, 1])
    """
    img = img[60:135, :, :]                          # 1. Crop
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)       # 2. YUV colour space
    img = cv2.GaussianBlur(img, (3, 3), 0)           # 3. Gaussian blur
    img = cv2.resize(img, (200, 66))                 # 4. Resize to 200×66
    img = img / 255.0                                # 5. Normalise to [0, 1]
    return img


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – BATCH GENERATOR (Section 6 of project spec)
# ══════════════════════════════════════════════════════════════════════════════

def batchGenerator(imagePaths, steerings, batchSize=100, isTraining=True):
    """
    Keras-compatible generator that yields (X_batch, y_batch) tuples
    indefinitely.  Keras's model.fit() will call next() on this generator
    for each training step.

    Why a generator instead of loading everything into RAM?
    -------------------------------------------------------
    The full dataset (5 laps × 2 directions × 3 cameras) can easily be
    several GB of images.  A generator loads and preprocesses only one
    batch at a time, keeping memory usage low.

    Parameters
    ----------
    imagePaths : list of str   – balanced image file paths
    steerings  : list of float – corresponding steering angles
    batchSize  : int           – number of samples per batch (default 100)
    isTraining : bool          – if True, apply random augmentation

    Yields
    ------
    (X, y) where X.shape = (batchSize, 66, 200, 3)  and  y.shape = (batchSize,)
    """
    while True:                                   # loop forever for Keras
        # Pick `batchSize` random indices WITHOUT replacement each iteration
        indices = random.sample(range(len(imagePaths)), batchSize)

        imgBatch      = []
        steeringBatch = []

        for i in indices:
            img      = mpimg.imread(imagePaths[i])   # load as RGB (uint8)
            steering = steerings[i]

            # Augment ONLY during training
            if isTraining:
                img, steering = randomAugment(img, steering)

            # Preprocess (same pipeline used at inference)
            img = preProcessing(img)

            imgBatch.append(img)
            steeringBatch.append(steering)

        yield np.array(imgBatch), np.array(steeringBatch)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – TRAIN / VALIDATION SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def prepareData(dataPath='data', testSize=0.2, display=False):
    """
    Full pipeline: load → balance → split.

    Returns
    -------
    xTrain, xVal : lists of image paths
    yTrain, yVal : lists of steering angles
    """
    imagePaths, steerings = loadData(dataPath)
    imagePaths, steerings = balanceData(imagePaths, steerings, display=display)

    xTrain, xVal, yTrain, yVal = train_test_split(
        imagePaths, steerings,
        test_size=testSize,
        random_state=42        # fixed seed → reproducible split
    )

    print(f'[dataPreprocessing] Training samples   : {len(xTrain)}')
    print(f'[dataPreprocessing] Validation samples : {len(xVal)}')

    return xTrain, xVal, yTrain, yVal


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – STANDALONE TEST (verify a few augmented images look right)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    DATA_PATH = 'data'

    xTrain, xVal, yTrain, yVal = prepareData(DATA_PATH, display=True)

    # Show 4 random augmented + preprocessed training images
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    fig.suptitle('Augmented → Preprocessed Training Samples')

    for col in range(4):
        idx = random.randint(0, len(xTrain) - 1)
        raw = mpimg.imread(xTrain[idx])

        augImg, augSteer = randomAugment(raw.copy(), yTrain[idx])
        preImg           = preProcessing(augImg)

        axes[0, col].imshow(raw)
        axes[0, col].set_title(f'Raw  steer={yTrain[idx]:.3f}')
        axes[0, col].axis('off')

        axes[1, col].imshow(preImg)
        axes[1, col].set_title(f'Processed steer={augSteer:.3f}')
        axes[1, col].axis('off')

    plt.tight_layout()
    plt.show()
