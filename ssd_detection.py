# # Object Detection
# Using a pre-trained model to detect objects in an image.

# # Imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import cv2
import time
import pafy

def is_url(path):
  try:
    result = urlparse(path)
    return result.scheme and result.netloc and result.path
  except:
    return False

# In the object_detection folder.
sys.path.append("..")

tf.compat.v1.flags.DEFINE_string('video', '0', 'Path to the video file.')
FLAGS = tf.compat.v1.flags.FLAGS
video = FLAGS.video

if is_url(video):
  videoPafy = pafy.new(video)
  video = videoPafy.getbest(preftype="mp4").url
elif video == '0' or video == '1':
  video = int(video)

cap = cv2.VideoCapture(video)
if not cap.isOpened():
  raise IOError('Can\'t open "{}"'.format(FLAGS.video))

# ## Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables

# detection model zoo(https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
 file_name = os.path.basename(file.name)
 if 'frozen_inference_graph.pb' in file_name:
   tar_file.extract(file, os.getcwd())

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Function to reshape image
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection
with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    start_time = time.time()
    fps = 0
    try:
      while True:
        ret, image_np = cap.read()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

        # Print objects to terminal
        i = 0
        while (i < len(np.squeeze(scores))):
          objNum = 0
          currentScore = np.squeeze(scores)[i]
          if currentScore >= 0.75:
            currentClasses = np.squeeze(classes).astype(np.int32)[i]
            objNum = objNum + 1
            print(category_index[currentClasses]["name"] + " {:.2f}%".format(currentScore*100))
          i = i + 1
        print("==========")

        end_time = time.time()
        fps = fps * 0.9 + 1 / (end_time - start_time) * 0.1
        start_time = end_time

        # Draw FPS in image
        frame_info = 'FPS: {0:.2f}'.format(fps)
        cv2.putText(image_np, frame_info, (10, image_np.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 200, 25), 1)

        #cv2.imshow('SSD object detection', cv2.resize(image_np, (1280, 720)))
        cv2.imshow('SSD object detection', image_np)

        key = cv2.waitKey(1) & 0xFF

        # Exit
        if key == ord('q'):
          break

        # Take screenshot
        if key == ord('s'):
          cv2.imwrite('frame_{}.jpg'.format(time.time()), image_np)

    finally:
      cap.release()
      cv2.destroyAllWindows()
