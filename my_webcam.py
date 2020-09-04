import cv2
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

# Initialize webcam feed
video = cv2.VideoCapture('output_5.mp4')

# video_writer = cv2.VideoWriter('output.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=20.0, frameSize=(1280,720))

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    if not ret:
        break
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # frame = cv2.resize(frame, (1280,720))
    # All the results have been drawn on the frame, so it's time to display it.
    # video_writer.write(frame)

    cv2.imshow('Object detector', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    time.sleep(0.01)

# Clean up
video.release()
# video_writer.release()

cv2.destroyAllWindows()