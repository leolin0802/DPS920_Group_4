"""
TestSimulation.py
DPS920 Final Project - Self-Driving Car Simulation
---------------------------------------------------------------------------
PURPOSE:
    Run the trained model in real-time inside the Udacity simulator.
    This script acts as a server that listens for telemetry data
    (speed + camera image) sent by the simulator over a socket connection,
    predicts a steering angle using the trained CNN, and sends back
    steering + throttle commands.

NOTE:
    This file was provided by the professor.  Do NOT modify the core
    socket/server logic.  Only the preprocessing function must match
    exactly what was used during training (dataPreprocessing.py).

USAGE:
    1. Make sure 'model.h5' is in the same folder as this script.
    2. Activate your conda environment:
           conda activate dps920
    3. Run this script FIRST:
           python TestSimulation.py
    4. THEN open the Udacity simulator and select Autonomous Mode.
    5. The car should start driving automatically.

TROUBLESHOOTING:
    - "model.h5 not found" → make sure you ran train.py first.
    - Car swerves off immediately → preprocessing mismatch; double-check
      the preProcessing() function below matches dataPreprocessing.py.
    - Connection refused → make sure port 4567 is not blocked by a firewall.
---------------------------------------------------------------------------
"""

import os
print('Setting Up ...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress TensorFlow C++ warnings

import socketio
import eventlet
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask
import base64
from io import BytesIO
from PIL import Image
import cv2

# Socket.IO server (handles real-time bidirectional communication with simulator)
sio = socketio.Server()
app = Flask(__name__)

# Maximum speed for the car (used to compute throttle)
# The throttle formula: throttle = 1.0 - speed/maxSpeed
# → full throttle when stopped, zero throttle when at max speed
maxSpeed = 10


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING  –  MUST MATCH dataPreprocessing.py EXACTLY
# ══════════════════════════════════════════════════════════════════════════════

def preProcessing(img):
    """
    Apply the same preprocessing pipeline used during training:
        1. Crop rows 60–135  (remove sky + car hood, keep road)
        2. Convert RGB → YUV  (NVIDIA model colour space)
        3. Gaussian blur  (reduce noise)
        4. Resize to 200×66  (NVIDIA model input size)
        5. Normalise to [0, 1]

    ⚠ This function is intentionally identical to the one in
      dataPreprocessing.py.  If you change one, change the other.
    """
    img = img[60:135, :, :]                       # 1. Crop
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)    # 2. YUV
    img = cv2.GaussianBlur(img, (3, 3), 0)        # 3. Blur
    img = cv2.resize(img, (200, 66))              # 4. Resize
    img = img / 255                               # 5. Normalise
    return img


# ══════════════════════════════════════════════════════════════════════════════
#  SOCKET EVENTS
# ══════════════════════════════════════════════════════════════════════════════

@sio.on('telemetry')
def telemetry(sid, data):
    """
    Called every frame by the simulator.

    The simulator sends:
        data['speed'] : current speed of the car (float as string)
        data['image'] : base64-encoded JPEG from the front camera

    We respond with steering angle + throttle via sendControl().
    """
    speed = float(data['speed'])

    # Decode the base64 image → PIL Image → numpy array (RGB uint8)
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)

    # Apply the same preprocessing pipeline as training
    image = preProcessing(image)

    # Add batch dimension: (66, 200, 3) → (1, 66, 200, 3)
    image = np.array([image])

    # Predict steering angle from the CNN
    steering = float(model.predict(image))

    # Throttle: proportional speed control
    # At speed=0 → throttle=1.0  (accelerate fully)
    # At speed=maxSpeed → throttle=0.0  (coast)
    throttle = 1.0 - speed / maxSpeed

    print(f'Throttle: {throttle:.3f} | Steering: {steering:.3f} | Speed: {speed:.1f}')
    sendControl(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    """
    Called once when the simulator first connects.
    We send a neutral command (steer=0, throttle=0) to initialise.
    """
    print('Simulator connected!')
    sendControl(0, 0)


def sendControl(steering, throttle):
    """
    Emit the 'steer' event back to the simulator with the computed
    steering angle and throttle value.
    """
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle':       throttle.__str__()
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Load the trained model from disk
    model = load_model('model.h5')
    print('Model loaded successfully. Waiting for simulator connection on port 4567...')

    # Wrap Flask app with Socket.IO middleware
    app = socketio.Middleware(sio, app)

    # Start the WSGI server on port 4567
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
