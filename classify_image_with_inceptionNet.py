# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

#This is my own file:
import functions



# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/Users/pam/Documents/dev/pretrainedNetworks/inception',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '', """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5, """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup, self.node_uidlookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name, node_id_to_uid

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

  def id_to_uid(self, node_id):
    if node_id not in self.node_uidlookup:
      return ''
    return self.node_uidlookup[node_id]

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


label_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
synonyms = [['plane', 'airliner', 'aircraft'],
            ['bird', 'hen', 'cock', 'partridge', 'grouse', 'finch', 'chicken', 'quail', ' robin', ' jay', 'chickadee', 'brambling', 'crane', 'ostrich', 'heron', 'pelican', 'stork', 'cockatoo', 'African gray', 'ptarmigan', 'partridge'],
            ['car', 'limo', 'Model T', 'convertible', 'taxi', 'minivan'],
            ['cat', 'kitten'],
            ['deer', 'stag', 'hind', 'fawn'],
            ['dog', 'terrier', 'Great Dane', 'Doberman', 'hound', 'corgi', 'collie', 'poodle', 'boxer', 'puppy'],
            ['horse', 'sorrel'],
            ['monkey', 'chimp', 'gorilla', 'baboon', 'capuchin', 'ape', 'primate'],
            ['ship', 'trimaran'],
            ['truck']]
planeMin = 0
planeMax = 15102894
birdMin = 1503060
birdMax = 1625563
carMin = 0
carMax = 15102894
catMin = 2120996
catMax = 2130926
deerMin = 2419796
deerMax = 2436646
dogMin = 2082790
dogMax = 2116739
horseMin = 2374148
horseMax = 2391618 # includes zebras
monkeyMin = 2480152 # excluding man etc!
monkeyMax = 2503126 # including teeny tiny monkeys and lemurs and gorillas?!
shipMin = 0
shipMax = 15102894
truckMin = 0
truckMax = 15102894
max = 2590
#max = 20

def checkSynonyms_words(labelIdx, human_string):
  for word in synonyms[labelIdx]:
    #print("Synonym: {}".format(word))
    if word in human_string:
      #print("Found")
      return True
  return False

def checkSynonyms(labelIdx, node_id, human_string):
  label = label_names[labelIdx]
  nodeNumber = node_id.replace("n", "")
  nodeNumber = int(nodeNumber)
  if label == 'plane' or label == 'airplane':
    return checkSynonyms_words(labelIdx, human_string)
    #if nodeNumber > planeMin and nodeNumber < planeMax:
    #  return True
  if label == 'bird':
    if nodeNumber > birdMin and nodeNumber < birdMax:
      return True
    return checkSynonyms_words(labelIdx, human_string)
  if label == 'car':
    return checkSynonyms_words(labelIdx, human_string)
    #if nodeNumber > carMin and nodeNumber < carMax:
    #  return True
  if label == 'cat':
    if nodeNumber > catMin and nodeNumber < catMax:
      return True
    return checkSynonyms_words(labelIdx, human_string)
  if label == 'dog':
    if nodeNumber > dogMin and nodeNumber < dogMax:
      return True
    return checkSynonyms_words(labelIdx, human_string)
  if label == 'deer':
    if nodeNumber > deerMin and nodeNumber < deerMax:
      return True
    return checkSynonyms_words(labelIdx, human_string)
  if label == 'horse':
    if nodeNumber > horseMin and nodeNumber < horseMax:
      return True
    return checkSynonyms_words(labelIdx, human_string)
  if label == 'monkey':
    if nodeNumber > monkeyMin and nodeNumber < monkeyMax:
      return True
    return checkSynonyms_words(labelIdx, human_string)
  if label == 'ship':
    return checkSynonyms_words(labelIdx, human_string)
    #if nodeNumber > shipMin and nodeNumber < shipMax:
    #  return True
  if label == 'truck':
    return checkSynonyms_words(labelIdx, human_string)
    #if nodeNumber > truckMin and nodeNumber < truckMax:
    #  return True

  return False

def get_RGB_from_file(filename, imgNum=0, width=96, height=96, channels=3):
  with open(filename, "rb") as f:
    allTheData = np.fromfile(f, 'u1')
  recordSize = (width * height * channels) + 1
  num_cases_per_batch = allTheData.shape[0] / recordSize
  #print("number of cases = {}".format(num_cases_per_batch))

  allTheData = allTheData.reshape(num_cases_per_batch, recordSize)
  data_labels = allTheData[:, 0].copy()
  data_array = allTheData[:, 1:].copy()

  label = data_labels[imgNum]
  pixel_depth = 8
  frame = data_array[imgNum]
  frame = frame * pixel_depth
  rgbframe = functions.planarYUV_2_planarRGB(frame, height, width)

  rgbframe = rgbframe.reshape(channels, height, width)
  rgbframe = np.swapaxes(rgbframe, 0, 1)
  rgbframe = np.swapaxes(rgbframe, 1, 2)

  return label, rgbframe




def run_inference_on_image(image):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  #image_tensor = tf.image.decode_jpeg(image_data)
  #image_tensor = image_tensor.get_tensor_by_name('DecodeJpeg/contents:0')
  #image_tensor = {'DecodeJpeg:0': image_tensor}
  #print("Shape is {}".format(image_tensor.get_shape))
  #image_array = np.array(image_tensor)

  # Getting RGB data out of my own files
  height = 96
  width = 96
  channels = 3
  pixel_depth = 8
  datadir = '/Volumes/LaCie/stl10_binary/constantQuant/refactored_anew/'
  datadir = '/Users/pam/Documents/data/stl10/smallDataset/'
  batchfileName = 'datasetstl10_binary'
  binfileNames = ['test_X.bin']
  datasetNames = ["yuv", "y_squv", "q10_f0", "q25_f0", "q37_f0", "q41_f0", "q46_f0", "q50_f0"]

  data_folders = []
  for binfileName in binfileNames:
    for name in datasetNames:
      data_folders.append(datadir + "/" + batchfileName + "_" + name + "/" + binfileName)

  filename = data_folders[0]
  print("The filename is {}".format(filename))
  image_array = []
  label = []
  for i in range(0, max):
    for j in range(0, len(data_folders)):
      idx = (i*len(data_folders)) + j
      labeli, image_arrayi = get_RGB_from_file(data_folders[j], imgNum=i)
      image_array.append(image_arrayi)
      label.append(labeli)
      #print("From file {}".format(data_folders[j]))
      #print("The label is {} and number is {}, i {}, j {}".format(label_names[labeli], idx, i, j))
      #print("Extracted the image data, shape: {}".format(image_array[idx].shape))


  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    #predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    totalScore = [0, 0, 0, 0, 0, 0, 0, 0]
    totalCorrects = [0, 0, 0, 0, 0, 0, 0, 0]
    totalPlacings = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0,max):
      if i % 100 == 0:
        print("Total scores:")
        for j in range(0, len(data_folders)):
          print("{}: corrects: {} score: {} placings: {}".format(data_folders[j], totalCorrects[j], totalScore[j], totalPlacings[j]))
      for j in range(0, len(data_folders)):
        idx = (i*len(data_folders)) + j
        predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': image_array[idx]})
        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        node_lookup = NodeLookup()

        top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
        #print("\n\nThe label is {} and {}".format(label_names[label[idx]], datasetNames[j]))
        correct = False
        outputString = "\n\nThe label is {} and {} \n".format(label_names[label[idx]], datasetNames[j])
        for myIdx, node_id in enumerate(top_k):
          totalScore[j] = totalScore[j] + predictions[node_id]
          human_string = node_lookup.id_to_string(node_id)
          uid = node_lookup.id_to_uid(node_id)
          #print("The node_id is: {} description: {}".format(uid, human_string))
          score = predictions[node_id]
          #correct = checkSynonyms_words(label[idx], human_string)
          correct = checkSynonyms(label[idx], uid, human_string)
          if correct:
            print("Correct in {} set! Label: {} and correct guess({}): {}, confidence {}".format(datasetNames[j], label_names[label[idx]], myIdx, human_string, score))
            totalCorrects[j] = totalCorrects[j] + 1
            totalPlacings[j] = totalPlacings[j] + myIdx
            break
          outputString += '%s (score = %.5f) \n' % (human_string, score)
          #print('%s (score = %.5f)' % (human_string, score))

        # not printing out the ones I expect to be crap for now
        if not correct:
          print(outputString)

    print("Total scores:")
    for j in range(0, len(data_folders)):
      print("{}: corrects: {} score: {} placings: {}".format(data_folders[j], totalCorrects[j], totalScore[j], totalPlacings[j]))


def run_inference_on_image_orig(image):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  maybe_download_and_extract()
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
  print(image)
  run_inference_on_image_orig(image)


if __name__ == '__main__':
  tf.app.run()
