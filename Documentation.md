
# DPS920 Group 4 – Project Documentation

---

## 1. Project Overview

This project implements an end-to-end self-driving car simulation using a CNN based on NVIDIA's autonomous driving architecture. The model takes images from the car's front camera and predicts a steering angle to keep the car on the road inside the Udacity simulator.

---

## 2. Team Contributions

**Mao-Sen** – Set up the GitHub repository, wrote all core Python scripts (`dataCollection.py`, `dataPreprocessing.py`, `train.py`, `TestSimulation.py`), and established the overall project structure.

**Vanshdeep Kaur** – Ran the full pipeline end-to-end, identified and resolved all dependency and environment issues, collected driving data, retrained the model to improve car stability, and contributed towards documentation.

**Alyson** – Wrote the main project documentation, identified bugs during testing, and helped verify the pipeline worked correctly from data collection through autonomous driving.

---

## 3. Approach

- **Data Collection:** The car was driven manually in Training Mode for approximately 5 laps forward and 5 laps in reverse on Track 1 using mouse steering for smooth steering labels.

- **Data Balancing:** The raw dataset is heavily biased toward a steering angle of 0°. Histogram-based balancing was used to remove over-represented bins and produce a more uniform distribution.<br>
Balancing gives the model more appropriate data to work with because when the data is steering 0° majority of the time, highest bin, it overshadows the few turns made by the car on the track. When the model gets trained on this data, it will learns incorrectly that it should also always go straight even if a turn is needed. It goes off the track. By removing samples from the over represented straight steering bins, the model will account for the loss of turns and going straight, finding the balance between both and learning to turn when necessary.
<br>

- **Data Augmentation:** Applied randomly to training images only — flipping (with steering negation), panning, brightness adjustment, and zooming.

- **Preprocessing:** Crop rows 60–135 → convert RGB to YUV → Gaussian blur → resize to 200×66 → normalize to [0, 1].<br>
Cropping the rows can remove the sky, trees, and distant horizon which are not important to tracking the road, so the model can focus on the road and remove unnecessary noise and processing.
Gaussian blur removes noise and sharp edges from leaves or shadows so that the model learns general curves and boundaries of the road rather than exact pixel details
<br>

- **Model:** NVIDIA end-to-end CNN — 5 convolutional layers, flatten, dropout (0.5), 4 dense layers (100 → 50 → 10 → 1). Loss: MSE. Optimizer: Adam.

- **Results:** The model successfully drives the car around Track 1, completing a full lap. Minor issues include occasional instability near dirt edges at the end of the lap, and difficulty navigating Track 2.

---

## 4. Challenges & Solutions

- **Python & TensorFlow version conflict** — TensorFlow does not support Python 3.12+. Recreated the virtual environment explicitly with Python 3.11 and pinned compatible package versions.

- **eventlet crash on Python 3.11** — `eventlet==0.25.1` throws a `TypeError` on Python 3.11. Upgraded to `eventlet==0.33.3`.

- **Flask/Werkzeug/Jinja2 conflicts** — `flask==1.1.2` was incompatible with newer Jinja2 and Werkzeug versions. Upgraded Flask to `2.0.3` and pinned Werkzeug to `2.0.3`.

- **Keras safe_mode restriction** — Newer Keras refuses to load models with Lambda layers by default. Added `safe_mode=False` to the `load_model()` call in `TestSimulation.py`.

- **Insufficient training data** — Batch size exceeded dataset size. Reduced `BATCH_SIZE` to 50 and collected additional driving data before retraining.

- **model.h5 version mismatch** — Model saved on one machine was incompatible with a different Keras version on another. Retrained from scratch locally.

---

## 5. How to Run

### Environment Setup

### MacOs
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

### Windows
```bash
# Create venv with Python 3.11
& "C:\...\Python311\python.exe" -m venv .venv
.venv\Scripts\activate

# Install dependencies in this order
pip install numpy==1.26.4
pip install tensorflow==2.13.0
pip install flask==2.0.3 werkzeug==2.0.3
pip install flask-socketio==5.3.4
pip install python-socketio==5.8.0 python-engineio==4.5.1
pip install eventlet==0.33.3
pip install opencv-python pandas matplotlib scikit-learn pillow imgaug
pip install scipy==1.11.4 --force-reinstall
```

### Training

```bash
python train.py
```

### Testing

```bash
python TestSimulation.py
# Then open simulator → Autonomous Mode → select track
```
