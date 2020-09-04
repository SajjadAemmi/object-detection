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

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'images/labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

detection_model = tf.saved_model.load(str('inference_graph/saved_model'))
model_fn = detection_model.signatures['serving_default']

# Initialize webcam feed
video = cv2.VideoCapture('videos/video_3.h264')
# video = cv2.VideoCapture(0)

# video_writer = cv2.VideoWriter('output_5.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30.0, frameSize=(1280,720))

count = 52

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    original_frame = frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    input_tensor = tf.convert_to_tensor(frame)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    
    output_dict = model_fn(input_tensor)
    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}

    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                frame.shape[0], frame.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.1,
                                        tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    vis_util.visualize_boxes_and_labels_on_image_array(
      frame,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      min_score_thresh=.7,
      max_boxes_to_draw=1,
      line_thickness=8)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)    
    
    # if np.max(output_dict['detection_scores']) > 0.7:
    #   index = np.argmax(output_dict['detection_scores'])
    #   class_id = output_dict['detection_classes'][index]
    #   text = category_index[class_id]['name']
    #   cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    # video_writer.write(frame)

    if cv2.waitKey(1) == ord('c'):
      cv2.imwrite(f'{count}.jpg', original_frame)
      count += 1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
# video_writer.release()
cv2.destroyAllWindows()