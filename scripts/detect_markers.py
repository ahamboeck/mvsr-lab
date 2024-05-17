# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Reshape, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2 as cv
import os

# %%
tf.__version__

# %%
# Load the ML model
model = tf.keras.models.load_model('../models/model.keras')

# %%
# Constants
img_shape_full = (300, 300, 1)

# %%
# Get webcam stream
cap = cv.VideoCapture(0)

# Set the resolution
cap.set(3, 640)

# %%
num_contours = 20
desired_dim = (300, 300)  # Example desired dimension

# %%
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for contour detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Find contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Process only up to `num_contours` (using Python slicing)
    for cnt in contours[:num_contours]:
        # Get the bounding rectangle for the contour
        x, y, w, h = cv.boundingRect(cnt)

        # Crop and resize the image
        cropped_image = gray[y:y + h, x:x + w]
        resized_image = cv.resize(cropped_image, desired_dim)


        # Draw the current contour on the original frame if the model predicts it as true
        if model.predict(np.array(resized_image).reshape(1, 300, 300, 1))[0][0] > 0.95:
            cv.drawContours(frame, [cnt], 0, (0, 255, 0), 3)
        

    # Display the frame with drawn contours
    cv.imshow('frame', frame)

    # Break the loop on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv.destroyAllWindows()


