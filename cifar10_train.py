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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cifar10
import cifar10_eval
import image2vid

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
#                           """Directory where to write event logs """
#                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir', '/Users/pam/Documents/data/CIFAR-10/test3/cifar10_train/train_yuv',
                           """Directory where to write event logs """
                           """and checkpoint.""")
#tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('max_steps', 30000, """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('mylog_dir', '/Users/pam/Documents/data/CIFAR-10/tutorial/',
                           """Directory where to write my logs """)


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()
    #quit()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference_switch(images, 4)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main_broken(argv=None):  # pylint: disable=unused-argument
  #cifar10.maybe_download_and_extract()
  saveFrames = (0, 2, 6)
  quants = (10, 25, 37, 50)
  #x264 = '../x264/x264'
  src_dir = os.path.join(FLAGS.data_dir, FLAGS.batches_dir)
  
  FLAGS.run_once = True
  #FLAGS.max_steps = 20
  #FLAGS.checkpoint_dir = '/Users/pam/Documents/data/CIFAR-10/tutorial/cifar10_train_mine/'
  #FLAGS.eval_dir = '/Users/pam/Documents/data/CIFAR-10/tutorial/cifar10_eval/'
  train_dir_base = FLAGS.train_dir
  data_dir_base = FLAGS.data_dir
  batches_dir_base = FLAGS.batches_dir
  eval_dir_base = FLAGS.eval_dir
  checkpoint_dir_base = FLAGS.checkpoint_dir

  myDatadirs = []
  myDatadirs = image2vid.generateDatasets('', src_dir, FLAGS.data_dir, '', x264, '', saveFrames, quants)
  #myDatadirs = ["yuv", "y_quv"]
  #for quant in quants:
  #  for idx, frame in enumerate(saveFrames):
  #    name = "q{}_f{}".format(quant, saveFrames[idx])
  #    myDatadirs.append(name)

  print(myDatadirs)
  logfile = os.path.join(FLAGS.mylog_dir, "log.txt")
  log = open(logfile, 'w')
  log.write("Here are the results \n")


  for dir in myDatadirs:
    myname = FLAGS.batches_dir + "_" + dir
    name = os.path.join(FLAGS.data_dir, myname)

    FLAGS.train_dir = train_dir_base  + "/train_" + dir
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    FLAGS.batches_dir = batches_dir_base  + "_" + dir
    FLAGS.checkpoint_dir = FLAGS.train_dir
    print("Train dir: {}".format(FLAGS.train_dir))
    print("Data dir: {}".format(FLAGS.data_dir))
    print("Batches dir: {}".format(FLAGS.batches_dir))
    print("Checkpoint dir: {}".format(FLAGS.checkpoint_dir))
    print("Eval dir: {}".format(FLAGS.eval_dir))
    train()
    for idx, datadir in enumerate(myDatadirs):
        FLAGS.batches_dir = batches_dir_base  + "_" + datadir
        precision = cifar10_eval.evaluate()
        print("Evaluating {} on {}-trained network: {}".format(datadir, dir, precision))
        log.write("Evaluating {} on {}-trained network: {}\n".format(datadir, dir, precision))
        log.flush()

def main(argv=None):  # pylint: disable=unused-argument
    #cifar10.maybe_download_and_extract()
    saveFrames = (0, 2, 3, 6)
    quants = (10, 25, 37, 41, 46, 50)
    bitrates = (200000, 100000, 50000, 35000, 20000, 10000)
    quants = (200000, 100000, 50000, 35000, 20000, 10000)
    x264 = '../x264/x264'
    src_dir = os.path.join(FLAGS.data_dir, FLAGS.batches_dir)

    FLAGS.run_once = True
    #FLAGS.max_steps = 20
    #FLAGS.checkpoint_dir = '/Users/pam/Documents/data/CIFAR-10/tutorial/cifar10_train_mine/'
    #FLAGS.eval_dir = '/Users/pam/Documents/data/CIFAR-10/tutorial/cifar10_eval/'
    train_dir_base = FLAGS.train_dir
    data_dir_base = FLAGS.data_dir
    batches_dir_base = FLAGS.batches_dir
    eval_dir_base = FLAGS.eval_dir
    checkpoint_dir_base = FLAGS.checkpoint_dir

    #myDatadirs = []
    #myDatadirs = image2vid.generateDatasets('', src_dir, FLAGS.data_dir, '', x264, '', saveFrames, quants)
    #datasetNames = ["yuv", "y_quv", "y_squv", "interlaced"]
    myDatadirs = ["yuv", "y_quv", "y_squv", "interlaced"]
    for quant in quants:
        for idx, frame in enumerate(saveFrames):
            name = "q{}_f{}".format(quant, saveFrames[idx])
            myDatadirs.append(name)

    # For a reduced test...
    myDatadirs = ["yuv", "q200000_f0", "q35000_f0", "q10000_f0"]

    print(myDatadirs)
    logfile = os.path.join(FLAGS.mylog_dir, "log.txt")
    log = open(logfile, 'w')
    log.write("Here are the results \n")


    my_data_dirs = []
    for dir in myDatadirs:
        myname = FLAGS.batches_dir + "_" + dir
        name = os.path.join(FLAGS.data_dir, myname)
        my_data_dirs.append(name)

    print("For testing: These are my datadirs: {}".format(my_data_dirs))

    for dir in myDatadirs:
        myname = FLAGS.batches_dir + "_" + dir
        name = os.path.join(FLAGS.data_dir, myname)

        FLAGS.train_dir = train_dir_base  + "/train_" + dir
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)
        FLAGS.batches_dir = batches_dir_base  + "_" + dir
        FLAGS.checkpoint_dir = FLAGS.train_dir
        print("Train dir: {}".format(FLAGS.train_dir))
        print("Data dir: {}".format(FLAGS.data_dir))
        print("Batches dir: {}".format(FLAGS.batches_dir))
        print("Checkpoint dir: {}".format(FLAGS.checkpoint_dir))
        print("Eval dir: {}".format(FLAGS.eval_dir))
        train()
        for idx, datadir in enumerate(myDatadirs):
            FLAGS.batches_dir = batches_dir_base  + "_" + datadir
            precision = cifar10_eval.evaluate()
            print("Evaluating {} on {}-trained network: {}".format(datadir, dir, precision))
            log.write("Evaluating {} on {}-trained network: {}\n".format(datadir, dir, precision))
            log.flush()


if __name__ == '__main__':
  tf.app.run()
