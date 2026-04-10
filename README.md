# DPS920 -Group4 Final Project – Self-Driving Car Simulation
**CNN-Based Steering Angle Prediction using the Udacity Simulator**

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Project File Structure](#2-project-file-structure)
3. [Environment Setup](#3-environment-setup)
   - Windows (Teammates 2 & 3)
   - macOS (Teammate 1)
4. [Step-by-Step Workflow](#4-step-by-step-workflow)
   - Step 1: Data Collection (Simulator)
   - Step 2: Inspect & Balance Data
   - Step 3: Preprocess & Augment
   - Step 4: Train the Model
   - Step 5: Test in Simulator
5. [Team Work Split](#5-team-work-split)
6. [Git Workflow](#6-git-workflow)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Project Overview

This project implements an end-to-end self-driving car neural network based on
NVIDIA's architecture. A Convolutional Neural Network (CNN) takes images from
the car's front camera as input and outputs a steering angle that keeps the car
on the road inside a Udacity simulation environment.

**Pipeline summary:**

```
Simulator (Training Mode)
       ↓
driving_log.csv + IMG/
       ↓
dataCollection.py   → load + balance steering distribution
       ↓
dataPreprocessing.py → augment (train only) + preprocess + batch
       ↓
train.py            → build NVIDIA CNN, train, save model.h5
       ↓
TestSimulation.py   → real-time inference in simulator (Autonomous Mode)
```

---

## 2. Project File Structure

```
project/
├── data/                      ← created by the simulator
│   ├── IMG/                   ← camera images (thousands of .jpg files)
│   └── driving_log.csv        ← steering + image path log
│
├── dataCollection.py          ← load CSV, plot histogram, balance dataset
├── dataPreprocessing.py       ← augmentation + preprocessing + batch generator
├── train.py                   ← build model, train, save model.h5
├── TestSimulation.py          ← inference server for simulator (given by prof)
│
├── model.h5                   ← saved trained model (generated after training)
├── training_plot.png          ← loss curve (generated after training)
└── README.md                  ← this file
```

---

## 3. Environment Setup

> ⚠ The package_list.txt is for **Windows (win-64)** only.
> macOS users must use pip to install equivalent packages.

---

### Windows Setup (Teammates 2 & 3)

#### Option A – Conda (Recommended, uses package_list.txt)

```bash
# 1. Open Anaconda Prompt
# 2. Navigate to your project folder
cd path\to\your\project

# 3. Create the environment from the provided package list
conda create --name dps920 --file package_list.txt

# 4. Activate it
conda activate dps920

# 5. Verify key packages
python -c "import tensorflow as tf; print(tf.__version__)"
# Expected: 2.3.0
```

#### Option B – pip (if conda fails)

```bash
conda create -n dps920 python=3.8
conda activate dps920
pip install tensorflow==2.3.0
pip install flask==1.1.2 flask-socketio==3.3.1
pip install python-socketio==4.2.1 python-engineio==3.8.2
pip install eventlet==0.25.1
pip install opencv-python numpy pandas matplotlib scikit-learn pillow
pip install imgaug
```

---

### macOS Setup (Teammate 1)

> TensorFlow 2.3.0 with GPU is not available on macOS. Use CPU version.

```bash
# 1. Open Terminal
# 2. Create a new conda environment
conda create -n dps920 python=3.8
conda activate dps920

# 3. Install packages via pip
pip install tensorflow==2.5.0          # closest CPU-compatible version on Mac
pip install flask==1.1.2
pip install flask-socketio==3.3.1
pip install "python-socketio==4.2.1"
pip install "python-engineio==3.8.2"
pip install eventlet==0.25.1
pip install opencv-python
pip install numpy pandas matplotlib scikit-learn pillow imgaug

# 4. Verify
python -c "import tensorflow as tf; print(tf.__version__)"
```

> **Mac note:** Training will be slower without a GPU. It is recommended that
> training be done on a Windows machine. The Mac can be used for data
> collection and running the inference server.

---

## 4. Step-by-Step Workflow

### Step 1: Download the Simulator & Collect Data

1. Download the Udacity simulator:
   - **Windows**: https://github.com/udacity/self-driving-car-sim
     (look for the Windows beta simulator release)
   - **Mac**: Search "udacity self-driving car simulator mac" as noted in the project spec

2. Extract the zip and run `beta_simulator.exe` (Windows) or the Mac equivalent.

3. In the configuration window (Figure 1 in spec):
   - Screen: 640×480
   - Graphics: Fastest
   - Click **Play**

4. Select **Training Mode**.

5. Click **RECORD** (top menu, red button) and choose your project's `data/` folder.

6. Drive the car using mouse steering:
   - **5 laps forward** on the left track
   - **5 laps in reverse** (turn the car around and drive the other way)
   - Use smooth mouse movements for cleaner steering labels

7. Click **RECORD** again to stop. The simulator saves:
   - `data/IMG/` – thousands of camera images
   - `data/driving_log.csv` – the steering log

---

### Step 2: Inspect & Balance Data

```bash
# From your project folder, with dps920 environment active:
python dataCollection.py
```

This will:
- Load `driving_log.csv` and print total sample count
- Show a **before** histogram of steering angle distribution
- Balance the dataset (remove over-represented near-zero angles)
- Show an **after** histogram (should look more uniform)

---

### Step 3: Verify Preprocessing

```bash
python dataPreprocessing.py
```

This will:
- Load and balance the data
- Show a grid of raw vs. augmented+preprocessed images
- Confirm the pipeline is working before training

---

### Step 4: Train the Model

```bash
python train.py
```

This will:
- Load, balance, and split the data (80% train / 20% validation)
- Build the NVIDIA CNN
- Print model summary
- Train for 10 epochs (adjust `EPOCHS` in the script if needed)
- Save `model.h5` (best checkpoint auto-saved during training)
- Show and save `training_plot.png`

**What to look for in the loss curves:**
- Both training and validation loss should decrease and converge
- If validation loss goes up while training loss goes down → overfitting
  → increase Dropout or reduce EPOCHS

---

### Step 5: Test in the Simulator

1. Make sure `model.h5` is in the same folder as `TestSimulation.py`.

2. With `dps920` environment active, run:
   ```bash
   python TestSimulation.py
   ```
   You should see: `Waiting for simulator connection on port 4567...`

3. Open the simulator → same settings → **Autonomous Mode**.

4. The car should start driving by itself. Record the screen for submission.

---

## 5. Team Work Split

The project is split into three parts based on the three Python scripts and their
associated tasks. Each person owns their script end-to-end (writing, testing, committing).

---

### 👤 Teammate 1 (macOS) – Data Collection & Balancing
**Owns: `dataCollection.py` + Simulator Data Collection**

**Responsibilities:**
1. Download the Udacity simulator (Mac version)
2. Collect the driving data:
   - At least 5 laps forward + 5 laps reverse on Track 1
   - Record to `data/` folder
3. Write and test `dataCollection.py`:
   - `loadData()` – reads driving_log.csv, extracts center image paths + steering
   - `plotHistogram()` – plots the steering distribution
   - `balanceData()` – removes over-represented bins
4. Verify the output by running `python dataCollection.py` and confirming:
   - Before histogram shows heavy spike at 0° (many straight-driving samples)
   - After histogram is more uniform
5. Share the `data/` folder with teammates (via USB, Google Drive, or Git LFS)
6. Commit `dataCollection.py` with clear commit messages

**Deliverable checklist:**
- [ ] `data/` folder collected and shared
- [ ] `dataCollection.py` committed and working
- [ ] Before/after histogram screenshots saved (for documentation)

**Key code to understand:**
```python
# In loadData(): why we use os.path.basename()
# The CSV stores the FULL path from the recording machine (different per computer).
# We strip the filename and rebuild the path locally so it works on any machine.
imagePaths = dataDF['center'].apply(
    lambda x: os.path.join(dataPath, 'IMG', os.path.basename(x.strip()))
).tolist()
```

---

### 👤 Teammate 2 (Windows) – Preprocessing & Augmentation
**Owns: `dataPreprocessing.py`**

**Responsibilities:**
1. Write and test `dataPreprocessing.py`:
   - `augmentFlip()` – horizontal flip + negate steering
   - `augmentPan()` – random horizontal/vertical shift
   - `augmentBrightness()` – random HSV brightness change
   - `augmentZoom()` – random centre zoom
   - `randomAugment()` – applies all four randomly
   - `preProcessing()` – crop → YUV → blur → resize → normalise
   - `batchGenerator()` – yields (X_batch, y_batch) for Keras
   - `prepareData()` – full pipeline: load → balance → split
2. Verify preprocessing matches what `TestSimulation.py` expects
3. Run `python dataPreprocessing.py` and confirm the image grid looks correct
4. Commit `dataPreprocessing.py`

**Deliverable checklist:**
- [ ] `dataPreprocessing.py` committed and working
- [ ] Can show augmented image samples visually
- [ ] `preProcessing()` matches `TestSimulation.py` exactly

**Key code to understand:**
```python
# Why augmentation uses isTraining=True/False in the generator:
# Augmentation is ONLY applied during training, not validation.
# Validation must use clean, unmodified images to give an honest
# evaluation of how the model performs on real data.

if isTraining:
    img, steering = randomAugment(img, steering)
```

```python
# Why we use a generator instead of loading all images at once:
# Thousands of images × 3 channels × 200×66 pixels = several GB.
# A generator loads one batch at a time → low memory usage.
while True:
    indices = random.sample(range(len(imagePaths)), batchSize)
    ...
    yield np.array(imgBatch), np.array(steeringBatch)
```

---

### 👤 Teammate 3 (Windows) – Model Training & Testing
**Owns: `train.py` + running `TestSimulation.py` + screen recording**

**Responsibilities:**
1. Write and test `train.py`:
   - `buildModel()` – implement the NVIDIA CNN architecture (Figure 7)
   - `trainModel()` – connect generators, run `model.fit()`, plot losses, save `model.h5`
2. Tune hyperparameters if training doesn't converge:
   - Adjust `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`
3. Run `TestSimulation.py` in the simulator:
   - Verify the car drives autonomously around the track
4. **Record the screen** showing the car completing the track
5. Commit `train.py`, `model.h5`, `training_plot.png`

**Deliverable checklist:**
- [ ] `train.py` committed and working
- [ ] `model.h5` saved and committed
- [ ] `training_plot.png` saved and committed
- [ ] Screen recording of car driving autonomously saved

**Key code to understand:**
```python
# The NVIDIA CNN architecture (from Figure 7):
# Input: (66, 200, 3) – matches preprocessing output
# 5 Conv layers: 3 with 5×5 kernels + stride 2, then 2 with 3×3 kernels
# 4 Dense layers: 100 → 50 → 10 → 1 (steering output)

model = Sequential([
    Lambda(lambda x: x / 0.5 - 1.0, input_shape=(66, 200, 3)),  # normalise
    Conv2D(24, (5,5), strides=(2,2), activation='elu'),
    Conv2D(36, (5,5), strides=(2,2), activation='elu'),
    Conv2D(48, (5,5), strides=(2,2), activation='elu'),
    Conv2D(64, (3,3), activation='elu'),
    Conv2D(64, (3,3), activation='elu'),
    Flatten(),
    Dropout(0.5),
    Dense(100, activation='elu'),
    Dense(50,  activation='elu'),
    Dense(10,  activation='elu'),
    Dense(1),   # linear output = steering angle
])
```

```python
# Why MSE (Mean Squared Error) for loss?
# Steering angle is a continuous number (regression, not classification).
# MSE penalises large errors more than small ones → encourages smooth driving.
model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
```

---

## 6. Git Workflow

```bash
# 1. One person creates the repo (e.g. on GitHub) and shares the link

# 2. Everyone clones it
git clone https://github.com/YOUR_TEAM/dps920-final-project.git
cd dps920-final-project

# 3. Each person works on their own branch
git checkout -b feature/data-collection      # Teammate 1
git checkout -b feature/preprocessing        # Teammate 2
git checkout -b feature/training             # Teammate 3

# 4. Commit with clear messages showing individual contributions
git add dataCollection.py
git commit -m "feat: implement loadData and balanceData functions"

git add dataPreprocessing.py
git commit -m "feat: add augmentation pipeline and batch generator"

git add train.py model.h5 training_plot.png
git commit -m "feat: train NVIDIA CNN, 10 epochs, val_loss=0.012"

# 5. Push your branch
git push origin feature/your-branch-name

# 6. Open a Pull Request → review → merge to main

# 7. Everyone pulls main before final submission
git checkout main
git pull origin main
```

**Add a `.gitignore` so you don't commit the data folder (it's too large):**

```
# .gitignore
data/
__pycache__/
*.pyc
.DS_Store
```

---

## 7. Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: cv2` | OpenCV not installed | `pip install opencv-python` |
| `No such file: driving_log.csv` | Wrong `DATA_PATH` | Change `DATA_PATH = 'data'` to your actual folder path |
| `Image file not found` | Absolute paths in CSV | The `os.path.basename()` fix in `loadData()` handles this |
| Car immediately drives off track | Preprocessing mismatch | Make sure `preProcessing()` in both files is identical |
| `ConnectionRefusedError` on port 4567 | Firewall blocking | Allow port 4567 in your firewall settings |
| Very high validation loss | Overfitting | Add more data, increase Dropout, reduce EPOCHS |
| Training very slow on Mac | No GPU | Expected – try reducing STEPS_PER_EPOCH and EPOCHS |
| `model.h5 not found` | Training not done | Run `train.py` first |

---

## Submission Checklist (Due April 13, 11:59 PM)

- [ ] `dataCollection.py` – working, committed
- [ ] `dataPreprocessing.py` – working, committed
- [ ] `train.py` – working, committed
- [ ] `TestSimulation.py` – working (from professor, with comments)
- [ ] `model.h5` – trained model, committed
- [ ] `training_plot.png` – loss curves, committed
- [ ] `README.md` – this file, committed
- [ ] Git repo with clear commit history from all 3 members
- [ ] Screen recording of car driving autonomously (MP4/MOV)
- [ ] Only ONE member submits on Blackboard/course portal
