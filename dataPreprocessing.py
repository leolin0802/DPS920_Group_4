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

# ============================================================================
# IMPORTS - External libraries we need for data processing
# ============================================================================

import os
# os: Provides operating system functions like file path handling
#     Not heavily used here but imported for potential path operations

import random
# random: Generates random numbers for augmentation decisions
#         Used to decide whether to flip, how much to pan, etc.

import numpy as np
# numpy: Handles numerical operations and array manipulations
#        Converts image batches to numpy arrays for TensorFlow

import matplotlib.pyplot as plt
# matplotlib.pyplot: Creates visualizations (histograms, image grids)
#                    Used in standalone test mode to verify augmentations

import matplotlib.image as mpimg
# matplotlib.image: Reads image files into numpy arrays
#                   Alternative to cv2.imread() that keeps RGB order

import cv2
# cv2 (OpenCV): Computer vision library for image processing
#               Handles flips, pans, brightness adjustments, zoom, resizing

from sklearn.model_selection import train_test_split
# train_test_split: Splits dataset into training and validation sets
#                   Ensures model is tested on unseen data

# Local module (must be in the same folder)
from dataCollection import loadData, balanceData
# loadData: Reads driving_log.csv and returns (image paths, steering angles)
# balanceData: Removes over-represented straight-driving samples (steering ≈ 0)
#              This prevents the model from learning to always go straight


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – AUGMENTATION FUNCTIONS
#  Applied RANDOMLY to training images only (not validation).
#  Each function takes (image, steering) and returns (augmented_image, new_steering).
#  
#  WHY AUGMENT? 
#  - Creates more training data from limited manual driving
#  - Helps model generalize to unseen situations
#  - Reduces overfitting (model memorizing specific track features)
# ══════════════════════════════════════════════════════════════════════════════

def augmentFlip(image, steering):
    """
    Horizontally flip the image with 50 % probability.
    When flipped the steering direction is reversed (multiply by -1).
    This doubles effective dataset size and balances left/right turns.
    
    WHY THIS WORKS:
    - Driving clockwise vs counter-clockwise produces opposite steering angles
    - Flipping the image simulates driving in the opposite direction
    - The steering angle must be negated because left becomes right
    
    VISUAL EXAMPLE:
    Original: Car sees left turn → steering = +0.5 (turn left)
    Flipped:  Image is mirrored → now appears as right turn → steering = -0.5
    """
    # random.random() returns float between 0.0 and 1.0
    # > 0.5 means exactly 50% probability of flipping
    if random.random() > 0.5:
        # cv2.flip(image, 1): 1 = horizontal flip (mirror left-right)
        # 0 would be vertical flip, -1 would be both axes
        image    = cv2.flip(image, 1)
        
        # Negate the steering angle because turning direction reverses
        # Example: +0.3 (right turn) becomes -0.3 (left turn after flip)
        steering = -steering
    return image, steering


def augmentPan(image, steering):
    """
    Randomly shift (pan) the image horizontally and vertically by up to
    10 % of image dimensions.  A horizontal pan simulates the car being
    off-centre, so we nudge the steering angle slightly.
    
    WHY THIS WORKS:
    - In real driving, the car isn't always perfectly centered in the lane
    - Panning simulates the car being left or right of center
    - The steering adjustment teaches the model to correct back to center
    
    MATHEMATICAL DETAIL:
    - tx: horizontal shift as pixels (positive = right shift)
    - steering adjustment: tx/w * 0.3 (scales shift to steering angle)
    - 0.3 is an empirically chosen scaling factor
    """
    # Get image dimensions: height (rows) and width (columns)
    h, w = image.shape[:2]  # shape[:2] ignores color channels (3)
    
    # Calculate random shifts:
    # (random.random() - 0.5) gives range [-0.5, 0.5]
    # Multiply by 0.2 gives range [-0.1, 0.1] = ±10% of dimension
    # Multiply by dimension (w or h) converts percentage to pixels
    tx   = w * (random.random() - 0.5) * 0.2   # Horizontal shift: ±10% of width
    ty   = h * (random.random() - 0.5) * 0.2   # Vertical shift: ±10% of height
    
    # Create affine transformation matrix for panning (translation only)
    # [[1, 0, tx],    This matrix moves the image:
    #  [0, 1, ty]]    - tx pixels right (negative = left)
    #                  - ty pixels down (negative = up)
    M    = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply the affine transformation (warp) to the image
    # warpAffine(src, M, dsize) where dsize is output dimensions
    image    = cv2.warpAffine(image, M, (w, h))
    
    # Adjust steering angle based on horizontal shift
    # tx/w: shift as fraction of image width (range -0.1 to 0.1)
    # Multiply by 0.3: reduces the correction (empirical tuning)
    # If car shifts right (positive tx), need to steer left (negative correction)
    steering = steering + tx / w * 0.3
    return image, steering


def augmentBrightness(image, steering):
    """
    Randomly adjust brightness by converting to HSV, scaling the V channel,
    then converting back to RGB.  This teaches the model to generalise across
    different lighting conditions (day / shadows / tunnel).
    
    WHY HSV COLOR SPACE?
    - HSV = Hue (color), Saturation (color intensity), Value (brightness)
    - Changing V channel affects brightness WITHOUT changing colors
    - RGB space would require complex math to maintain colors
    
    WHY BRIGHTNESS AUGMENTATION?
    - Real tracks have shadows, sunny spots, tunnels
    - Simulator lighting can vary based on graphics settings
    - Model learns to focus on road features, not lighting conditions
    """
    # Convert from RGB to HSV color space
    # cv2.COLOR_RGB2HSV: OpenCV expects RGB, not BGR
    # astype(np.float32): Convert to float for safe multiplication without overflow
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Generate random brightness factor between 0.5 (darker) and 1.5 (brighter)
    # random.random() returns [0.0, 1.0], so 0.5 + random gives [0.5, 1.5]
    factor = 0.5 + random.random()
    
    # Scale the Value (brightness) channel (channel index 2 in HSV)
    # HSV channels: [0]=Hue, [1]=Saturation, [2]=Value
    # np.clip() ensures values stay within valid 0-255 range
    image[:, :, 2] = np.clip(image[:, :, 2] * factor, 0, 255)
    
    # Convert back to RGB color space
    # astype(np.uint8): Convert back to 8-bit integer (0-255 range)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Note: steering angle doesn't change with brightness
    return image, steering


def augmentZoom(image, steering):
    """
    Randomly zoom into the centre of the image by up to 30 %.
    The zoomed region is resized back to the original dimensions so
    downstream preprocessing receives a consistent shape.
    
    WHY ZOOM AUGMENTATION?
    - Simulates the car being closer or further from obstacles
    - Changes the perceived scale of the road
    - Forces model to focus on road shape, not absolute distances
    
    HOW IT WORKS:
    1. Calculate zoom factor (1.0 = no zoom, 1.3 = 30% zoom)
    2. Calculate new smaller dimensions based on zoom factor
    3. Extract center region of new dimensions
    4. Resize back to original dimensions
    """
    h, w = image.shape[:2]  # Get original height and width
    
    # Generate random zoom factor between 1.0 and 1.3
    # 1.0 = no zoom, 1.3 = zoom in by 30% (image becomes 30% larger in apparent size)
    factor = 1 + random.random() * 0.3  # random() * 0.3 = [0, 0.3], then +1 = [1.0, 1.3]
    
    # Calculate new dimensions after zooming
    # If factor=1.3, new_h = h/1.3 ≈ 77% of original height
    # We're extracting a smaller region, which makes objects appear larger (zoomed in)
    new_h  = int(h / factor)
    new_w  = int(w / factor)
    
    # Calculate top-left corner (y1, x1) to extract center region
    # (h - new_h) // 2: This centers the crop region
    # Example: h=160, new_h=120 → (40)//2 = 20 pixels from top
    y1     = (h - new_h) // 2
    x1     = (w - new_w) // 2
    
    # Extract (crop) the center region of the image
    # This creates a smaller image (e.g., from 160x320 to 120x240)
    image  = image[y1:y1 + new_h, x1:x1 + new_w]
    
    # Resize the cropped region BACK to original dimensions (e.g., 160x320)
    # This makes everything appear 30% larger (zoom effect)
    image  = cv2.resize(image, (w, h))
    
    # Note: steering angle doesn't need adjustment for zoom
    return image, steering


def randomAugment(image, steering):
    """
    Apply each augmentation function randomly and independently.
    Called ONLY on training samples inside the batch generator.
    
    WHY CHAIN THEM?
    - Real driving has multiple variations simultaneously
    - Example: Car could be off-center (pan), in shadow (brightness), 
               and driving a different direction (flip) all at once
    - Combined augmentations create highly diverse training data
    
    ORDER MATTERS? 
    - Order is somewhat arbitrary but should be consistent
    - Flip is first because it's independent of other augmentations
    - Pan, brightness, zoom can be applied in any order
    """
    # Apply each augmentation in sequence
    # Each function returns (modified_image, possibly_modified_steering)
    # We pass the results to the next function (function composition)
    image, steering = augmentFlip(image, steering)
    image, steering = augmentPan(image, steering)
    image, steering = augmentBrightness(image, steering)
    image, steering = augmentZoom(image, steering)
    
    # Return the fully augmented image and adjusted steering angle
    return image, steering


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – PREPROCESSING PIPELINE
#  These exact steps are used BOTH during training and at inference time
#  (TestSimulation.py already contains the same pipeline – keep them in sync!)
#  
#  CRITICAL: If this doesn't match TestSimulation.py EXACTLY, the car will
#            fail to drive properly because training data format ≠ inference format
# ══════════════════════════════════════════════════════════════════════════════

def preProcessing(img):
    """
    Apply the mandatory preprocessing steps from the project spec (Section 5):

    1. Crop  – remove sky + hood (rows 60–135) to focus on the road.
    2. YUV   – convert colour space as required by the NVIDIA model.
    3. Blur  – Gaussian blur to reduce noise.
    4. Resize – to 200×66 as expected by the NVIDIA architecture.
    5. Normalise – scale pixel values to [0, 1].

    WHY EACH STEP?
    
    1. CROP:
       - Top 60 rows: Sky, trees, buildings (distracting, don't affect steering)
       - Bottom rows (135+): Car hood (always present, provides no useful info)
       - Remaining 75 rows focus purely on the road ahead
    
    2. YUV COLOR SPACE:
       - NVIDIA's paper used YUV (not RGB)
       - Y channel = luminance (brightness) - most important for road detection
       - U and V = color information (less important but still useful)
       - Separating brightness from color helps in varying lighting conditions
    
    3. GAUSSIAN BLUR:
       - Reduces high-frequency noise (camera sensor noise, texture details)
       - Smooths out small imperfections in road markings
       - Helps model focus on overall road structure, not pixel-level details
    
    4. RESIZE to 200×66:
       - NVIDIA architecture expects this exact input size
       - Reduces computational requirements (fewer pixels to process)
       - Maintains aspect ratio (original crop ~75x320 → resized to 66x200)
    
    5. NORMALIZE to [0, 1]:
       - Neural networks train better with normalized inputs
       - Prevents satuation of activation functions
       - Helps gradient descent converge faster

    Parameters
    ----------
    img : np.ndarray  (H × W × 3, RGB uint8) - Raw image from simulator

    Returns
    -------
    img : np.ndarray  (66 × 200 × 3, float32 in [0, 1]) - Ready for neural network
    """
    # 1. CROP: Remove sky (rows 0-60) and car hood (rows 135+)
    #    Keep only rows 60 to 135 (75 rows of road-focused content)
    #    :, : means keep all columns and all color channels
    img = img[60:135, :, :]
    
    # 2. YUV COLOR SPACE CONVERSION
    #    RGB (Red-Green-Blue) → YUV (Luminance-Chrominance)
    #    cv2.COLOR_RGB2YUV: Input is RGB (OpenCV normally uses BGR, so specify RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    # 3. GAUSSIAN BLUR
    #    (3, 3) = kernel size (3x3 pixel window for averaging)
    #    0 = sigma (standard deviation) - 0 means auto-calculate from kernel size
    #    Blur helps remove noise and high-frequency artifacts
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 4. RESIZE to NVIDIA expected dimensions
    #    (200, 66) = width, height (note: width first, then height!)
    #    Original crop ~75x320 → resized to 66x200
    #    Interpolation method defaults to bilinear (good balance speed/quality)
    img = cv2.resize(img, (200, 66))
    
    # 5. NORMALIZE to [0, 1] range
    #    Original pixel values: 0 to 255 (uint8 integers)
    #    After division: 0.0 to 1.0 (float32 decimals)
    #    Note: Some implementations use [-0.5, 0.5] by subtracting 0.5 after division
    img = img / 255.0
    
    # Return the fully preprocessed image ready for the neural network
    return img


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – BATCH GENERATOR (Section 6 of project spec)
#  
#  WHAT IS A GENERATOR?
#  - A Python function that uses 'yield' instead of 'return'
#  - It produces values one at a time, pausing between yields
#  - Can run indefinitely (while True loop)
#  
#  WHY USE A GENERATOR FOR TRAINING?
#  - Prevents out-of-memory errors (RAM overflow)
#  - Example: 10,000 images × 66×200×3 pixels × 4 bytes = 1.58 GB just for images
#  - With augmentation, memory usage would be even higher
#  - Generator loads only batchSize images at a time (e.g., 100 images = 16 MB)
#  
#  HOW KERAS USES THIS:
#  - model.fit(generator, steps_per_epoch=...)
#  - Keras calls next(generator) repeatedly until steps_per_epoch is reached
#  - Each call yields one batch of (images, steering_angles)
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
                                – if False, only preprocess (no augmentation)

    Yields
    ------
    (X, y) where X.shape = (batchSize, 66, 200, 3)  and  y.shape = (batchSize,)
    
    X: Batch of preprocessed images (ready for neural network)
    y: Batch of steering angles (target values for training)
    """
    
    # while True: This creates an infinite loop
    # The generator will keep yielding batches forever
    # Keras will stop when steps_per_epoch is reached in model.fit()
    while True:
        
        # Select random indices for this batch WITHOUT replacement
        # random.sample(population, k) picks k unique random elements
        # range(len(imagePaths)) creates list of all indices [0, 1, 2, ...]
        # batchSize determines how many unique indices to pick
        # Result: list of batchSize random indices (all different)
        indices = random.sample(range(len(imagePaths)), batchSize)
        
        # Initialize empty lists to store this batch's data
        imgBatch = []      # Will hold batchSize images
        steeringBatch = [] # Will hold batchSize steering angles
        
        # Loop through each randomly selected index
        for i in indices:
            # Load image from disk using matplotlib.image.imread()
            # imread() automatically returns RGB format (unlike cv2.imread which returns BGR)
            # Returns numpy array of shape (height, width, 3) with uint8 values (0-255)
            img = mpimg.imread(imagePaths[i])
            
            # Get the corresponding steering angle for this image
            steering = steerings[i]
            
            # Apply random augmentation ONLY during training
            # isTraining=True: Called from train.py during model fitting
            # isTraining=False: Called from train.py during validation (honest evaluation)
            if isTraining:
                img, steering = randomAugment(img, steering)
            
            # Apply preprocessing pipeline (crop → YUV → blur → resize → normalize)
            # This MUST be identical to what TestSimulation.py uses at inference time
            img = preProcessing(img)
            
            # Add processed image and steering to batch lists
            imgBatch.append(img)
            steeringBatch.append(steering)
        
        # Convert Python lists to numpy arrays (required by TensorFlow/Keras)
        # np.array(imgBatch): shape becomes (batchSize, 66, 200, 3)
        # np.array(steeringBatch): shape becomes (batchSize,)
        # 'yield' returns this batch and PAUSES the function here
        # Next call to generator() resumes from this point
        yield np.array(imgBatch), np.array(steeringBatch)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – TRAIN / VALIDATION SPLIT
#  
#  WHY SPLIT DATA?
#  - Training set: Used to update model weights (learn patterns)
#  - Validation set: Used to evaluate model on unseen data (detect overfitting)
#  - Test set: Used only once at the end (not used here - simulator is final test)
#  
#  TYPICAL SPLIT RATIOS:
#  - 80% training, 20% validation (used here)
#  - Some use 70/30 or 80/20 depending on dataset size
# ══════════════════════════════════════════════════════════════════════════════

def prepareData(dataPath='data', testSize=0.2, display=False):
    """
    Full pipeline: load → balance → split.

    Returns
    -------
    xTrain, xVal : lists of image paths
    yTrain, yVal : lists of steering angles
    
    Parameters:
    - dataPath: Folder containing driving_log.csv and IMG/ subfolder
    - testSize: Proportion of data to use for validation (0.2 = 20%)
    - display: Whether to show histogram (passed to balanceData)
    """
    
    # STEP 1: Load raw data from CSV file
    # loadData() reads driving_log.csv and returns:
    #   - imagePaths: List of full paths to center camera images
    #   - steerings: List of corresponding steering angles
    imagePaths, steerings = loadData(dataPath)
    
    # STEP 2: Balance dataset (remove excessive straight-driving samples)
    # Without balancing, model learns to always go straight (steering ≈ 0)
    # balanceData() removes some near-zero steering angles
    imagePaths, steerings = balanceData(imagePaths, steerings, display=display)
    
    # STEP 3: Split into training and validation sets
    # train_test_split from scikit-learn:
    #   - Randomly shuffles data (important to avoid temporal correlations)
    #   - Splits according to testSize (0.2 = 20% validation, 80% training)
    #   - random_state=42: Fixed seed ensures reproducible splits
    #     (42 is arbitrary - any number works, but keep consistent)
    xTrain, xVal, yTrain, yVal = train_test_split(
        imagePaths, steerings,
        test_size=testSize,
        random_state=42        # fixed seed → reproducible split
    )
    
    # Print dataset statistics to console
    # This helps verify the split worked as expected
    print(f'[dataPreprocessing] Training samples   : {len(xTrain)}')
    print(f'[dataPreprocessing] Validation samples : {len(xVal)}')
    
    # Return the four datasets
    # x = features (image paths), y = labels (steering angles)
    return xTrain, xVal, yTrain, yVal


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – STANDALONE TEST (verify a few augmented images look right)
#  
#  This code runs ONLY when you execute:
#      python dataPreprocessing.py
#  
#  It does NOT run when imported by train.py (because of if __name__ == '__main__')
#  
#  PURPOSE OF THIS TEST:
#  - Verify augmentation functions work correctly
#  - Visually inspect preprocessing output
#  - Catch errors early before training
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Define where the data is stored
    # This must match the folder name created by the simulator
    DATA_PATH = 'data'
    
    # Load and prepare the data (load → balance → split)
    # display=True shows the steering angle histogram
    xTrain, xVal, yTrain, yVal = prepareData(DATA_PATH, display=True)
    
    # Create a figure with 2 rows and 4 columns of subplots
    # figsize=(16, 6): 16 inches wide, 6 inches tall
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    
    # Set a title for the entire figure
    fig.suptitle('Augmented → Preprocessed Training Samples')
    
    # Loop through 4 columns (create 4 example images)
    for col in range(4):
        # Select a random image index from training set
        idx = random.randint(0, len(xTrain) - 1)
        
        # Load the raw image from disk (RGB format)
        raw = mpimg.imread(xTrain[idx])
        
        # Apply augmentation to a COPY of the raw image
        # raw.copy() ensures we don't modify the original
        # yTrain[idx] is the steering angle for this image
        augImg, augSteer = randomAugment(raw.copy(), yTrain[idx])
        
        # Apply preprocessing to the augmented image
        # This produces the final format that will go into the neural network
        preImg = preProcessing(augImg)
        
        # ROW 0 (top row): Show raw (original) images
        axes[0, col].imshow(raw)
        axes[0, col].set_title(f'Raw  steer={yTrain[idx]:.3f}')
        axes[0, col].axis('off')  # Turn off axis ticks/labels
        
        # ROW 1 (bottom row): Show preprocessed images
        # Note: preImg is in YUV color space and normalized, so display may look odd
        # This is expected - we're just checking that preprocessing completed
        axes[1, col].imshow(preImg)
        axes[1, col].set_title(f'Processed steer={augSteer:.3f}')
        axes[1, col].axis('off')
    
    # Adjust spacing between subplots to prevent overlap
    plt.tight_layout()
    
    # Display the figure on screen
    # Execution will pause here until user closes the window
    plt.show()
