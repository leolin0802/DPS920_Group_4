"""
dataCollection.py
DPS920 Final Project - Self-Driving Car Simulation
---------------------------------------------------------------------------
PURPOSE:
    Load the raw driving data recorded from the Udacity simulator,
    inspect the steering-angle distribution, balance the dataset by
    removing over-represented bins, and save the balanced image paths
    + steering values so the next script (dataPreprocessing.py) can
    pick them up.

USAGE:
    python dataCollection.py

EXPECTED INPUTS:
    data/driving_log.csv   – CSV produced by the simulator
    data/IMG/              – folder of camera images produced by the simulator

OUTPUT:
    Prints dataset statistics and shows a before/after histogram.
    Saves nothing to disk by itself; the balanced lists are imported by
    dataPreprocessing.py via the helper function at the bottom.
---------------------------------------------------------------------------
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

# ── 1. LOAD THE CSV ─────────────────────────────────────────────────────────

def loadData(dataPath):
    """
    Read driving_log.csv and return (imagePaths, steerings) as two lists.

    The CSV columns are:
        Center, Left, Right, Steering, Throttle, Brake, Speed

    We only need the Center image column and the Steering column,
    exactly as specified in the project instructions.

    Parameters
    ----------
    dataPath : str
        Path to the folder that contains driving_log.csv and the IMG sub-folder.

    Returns
    -------
    imagePaths : list of str
    steerings  : list of float
    """
    # Read CSV - no header row in some simulator versions, so we name columns manually
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    dataDF  = pd.read_csv(os.path.join(dataPath, 'driving_log.csv'), names=columns)

    # The CSV stores full absolute paths from the recording machine.
    # We extract just the filename and rebuild the path relative to THIS machine.
    # e.g.  "C:\\Users\\...\\IMG\\center_2021_....jpg"  →  "data/IMG/center_2021_....jpg"
    imagePaths = dataDF['center'].apply(
        lambda x: os.path.join(dataPath, 'IMG', os.path.basename(x.strip()))
    ).tolist()

    steerings = dataDF['steering'].tolist()

    print(f'[dataCollection] Total samples loaded : {len(imagePaths)}')
    return imagePaths, steerings


# ── 2. VISUALISE THE RAW DISTRIBUTION ───────────────────────────────────────

def plotHistogram(steerings, title='Steering Angle Distribution', bins=25):
    """
    Plot a histogram of steering angle values.
    A well-balanced dataset should look roughly uniform across all bins,
    similar to Figure 5 in the project instructions.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(steerings, bins=bins, color='steelblue', edgecolor='white')
    plt.title(title)
    plt.xlabel('Steering Angle')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


# ── 3. BALANCE THE DATASET ───────────────────────────────────────────────────

def balanceData(imagePaths, steerings, display=True):
    """
    Remove excess samples from over-represented steering-angle bins so
    the network does not become biased toward driving straight (angle ≈ 0).

    How it works
    ------------
    1. Divide the steering range [-1, 1] into `nBins` equal-width buckets.
    2. Compute a threshold  = average count per bin (with a cap so we don't
       remove too aggressively).
    3. For every bin that has MORE samples than the threshold, randomly drop
       the extras.

    Parameters
    ----------
    imagePaths : list of str
    steerings  : list of float
    display    : bool  – show before/after histograms if True

    Returns
    -------
    imagePaths : list  (balanced)
    steerings  : list  (balanced)
    """
    nBins     = 25
    samplesPerBin, binEdges = np.histogram(steerings, nBins)
    # Cap threshold at 1000 so we keep enough data even in large datasets
    threshold = min(int(np.mean(samplesPerBin) * 1.2), 1000)

    print(f'[dataCollection] Balance threshold per bin: {threshold}')

    removeIndices = []
    for i in range(nBins):
        binIndices = []   # indices of samples that fall in bin i
        for j, angle in enumerate(steerings):
            if binEdges[i] <= angle < binEdges[i + 1]:
                binIndices.append(j)
        # Shuffle so we drop a random subset, not always the same ones
        binIndices = shuffle(binIndices)
        removeIndices += binIndices[threshold:]   # keep only `threshold` samples

    print(f'[dataCollection] Samples removed during balancing: {len(removeIndices)}')

    # Delete the flagged indices
    imagePaths = [x for i, x in enumerate(imagePaths) if i not in removeIndices]
    steerings  = [x for i, x in enumerate(steerings)  if i not in removeIndices]

    print(f'[dataCollection] Samples after balancing         : {len(imagePaths)}')

    if display:
        plotHistogram(steerings, title='Steering Angle Distribution (After Balancing)')

    return imagePaths, steerings


# ── 4. MAIN (run standalone to inspect data) ─────────────────────────────────

if __name__ == '__main__':
    DATA_PATH = 'data'   # ← change if your simulator data lives elsewhere

    imagePaths, steerings = loadData(DATA_PATH)
    plotHistogram(steerings, title='Steering Angle Distribution (Before Balancing)')
    imagePaths, steerings = balanceData(imagePaths, steerings, display=True)
    print('[dataCollection] Done. Import loadData + balanceData from this module in dataPreprocessing.py')
