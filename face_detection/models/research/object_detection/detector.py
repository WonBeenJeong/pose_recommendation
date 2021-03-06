import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image



from utils import label_map_util

from utils import visualization_utils as vis_util

#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# download model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())



# load a model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#print(label_map)

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#print("--------------------------------------")
#print(categories)
# id: 1 ; name: person

category_index = label_map_util.create_category_index(categories)
#print("---------------------------------------")
#print(category_index)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  img = np.array(image.getdata())
  if img.size == im_height * im_width * 3:
    return img.reshape((im_height, im_width, 3)).astype(np.uint8)
  else:
    return np.zeros(1)


PATH_TO_TEST_IMAGES_DIR = '/home/tianli/face_detection/raw_data/' + sys.argv[1] + "/"
TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR) 
RESULT_ONE_PERSON_PATH = '/home/tianli/face_detection/results_one_person/' + sys.argv[1]
os.mkdir(RESULT_ONE_PERSON_PATH) # the images in this foler contain the bounding box

# the images in this folder are what we want
os.mkdir("/home/tianli/face_detection/results_one_person/original/" + sys.argv[1])

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    id = 1
    for image_path in TEST_IMAGE_PATHS:
      image_path = PATH_TO_TEST_IMAGES_DIR + image_path
      print("About to read in image: " + image_path + "\n")
      try:
      	image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      	image_np = load_image_into_numpy_array(image)
      	original_img = load_image_into_numpy_array(image)
      except:
	continue
      else:
      	if image_np.size == 1:
	  continue
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      	image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      	(boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      #print(np.squeeze(classes))
      #print(np.squeeze(boxes))
      # Visualization of the results of a detection.
      	if_one_person = vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      #plt.figure(figsize=IMAGE_SIZE)
      #plt.imsave("/home/tianli/face_detection/results/" + sys.argv[1] + "/"  + str(id) + ".jpg", image_np)
      	if if_one_person == 1:
	  plt.imsave("/home/tianli/face_detection/results_one_person/original/" + sys.argv[1] + "/" + str(id)+".jpg", original_img)
      	  plt.imsave(RESULT_ONE_PERSON_PATH + "/" + str(id) +".jpg", image_np)
	
        id += 1
