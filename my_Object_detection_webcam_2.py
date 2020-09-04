import cv2
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

threshold = 0.6

def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

def name(id):
  if id == 0:
    return "drink_pet_small"
  elif id == 1:
    return "drink_pet_large"
  elif id == 2:
    return "drink_aluminum"
  elif id == 3:
    return "detergents"
  elif id == 4:
    return "paper_large"
    

model_path = "tflite_tf2/model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
  
# Initialize webcam feed
video = cv2.VideoCapture('videos/video_2.h264')

# video_writer = cv2.VideoWriter('output_3.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30.0, frameSize=(1280,720))

while(True):
    ret, frame = video.read()
    if not ret:
      break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    h, w, _ = frame.shape

    input_data = cv2.resize(frame, (300, 300))
    input_data = input_data.astype(np.float32)
    input_data = input_data / 255.0
    input_data = np.reshape(input_data, (1, 300, 300, 3))
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(10):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)

    for obj in results:
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * w)
        xmax = int(xmax * w)
        ymin = int(ymin * h)
        ymax = int(ymax * h)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 3)
        print(obj['class_id'], obj['score'])

        cv2.putText(frame, name(obj['class_id']), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    # video_writer.write(frame)

    if cv2.waitKey(1) == ord('q'): # Press 'q' to quit
        break

# Clean up
video.release()
# video_writer.release()
cv2.destroyAllWindows()