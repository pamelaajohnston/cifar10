
"""A binary to train CIFAR-10 using multiple GPU's with synchronous updates.

Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

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
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10
import cifar10_eval
#import image2vid

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 30000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('mylog_dir', '/Users/pam/Documents/data/CIFAR-10/tutorial/',
                           """Directory where to write my logs """)
tf.app.flags.DEFINE_integer('network_architecture', 1,
                            """The number of the network architecture to use (inference function) """)

tf.app.flags.DEFINE_string('single_dir', '_yuv',
                           """For defining the single directory """)


def tower_loss(scope):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # Get images and labels for CIFAR-10.
  images, labels = cifar10.distorted_inputs()

  # Build inference Graph.
  logits = cifar10.inference_switch(images, FLAGS.network_architecture)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = cifar10.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(loss_name +' (raw)', l)
    tf.scalar_summary(loss_name, loss_averages.average(l))

  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    #print("The global step is {}".format(global_step))

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)
    decay_steps = 1500

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    cifar10.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(lr)
    #opt = tf.train.AdamOptimizer(lr)

    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
          # Calculate the loss for one tower of the CIFAR model. This function
          # constructs the entire CIFAR model but shares the variables across
          # all towers.
          loss = tower_loss(scope)

          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()

          # Retain the summaries from the final tower.
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          # Calculate the gradients for the batch of data on this CIFAR tower.
          grads = opt.compute_gradients(loss)

          # Keep track of the gradients across all towers.
          tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.scalar_summary('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.histogram_summary(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.histogram_summary(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.merge_summary(summaries)

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=gpu_options))
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
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

#      # evaluate on the test data periodically
#      if step % 10 == 0:
#        print("Evaluation time......!")
#        images_test, labels_test = cifar10.inputs(eval_data=True)
#        logits_test = cifar10.inference_switch(images_test, FLAGS.network_architecture)
#        top_k_op = tf.nn.in_top_k(logits_test, labels_test, 1)
#        num_iter = int(FLAGS.num_examples / FLAGS.batch_size)
#        print("Number of iterations = {}".format(num_iter))
#        true_count = 0  # Counts the number of correct predictions.
#        total_sample_count = num_iter * FLAGS.batch_size
#        print("total_sample_count = {}".format(total_sample_count))
#        step = 0
#        while step < num_iter:
#            print("Step is {}".format(step))
#            predictions = sess.run([top_k_op])
#            true_count += np.sum(predictions)
#            step += 1
#        # Compute precision @ 1.
#        precision = true_count / total_sample_count
#        print('%s: precision @ 1 = %.5f' % (datetime.now(), precision))


def main_orig(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

def main_testall(argv=None):  # pylint: disable=unused-argument
    #cifar10.maybe_download_and_extract()
    saveFrames = (0, 2, 3, 6)
    quants = (10, 25, 37, 41, 46, 50)
    bitrates = (200000, 100000, 50000, 35000, 20000, 10000)
    #quants = (200000, 100000, 50000, 35000, 20000, 10000)
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

def main_justFirstFrame(argv=None):  # pylint: disable=unused-argument
    #cifar10.maybe_download_and_extract()
    saveFrames = (0,)
    quants = (10, 25, 37, 41, 46, 50)
    #bitrates = (200000, 100000, 50000, 35000, 20000, 10000)
    #quants = (200000, 100000, 50000, 35000, 20000, 10000)
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
        print("Prelearned checkpoint file: {}".format(FLAGS.prelearned_checkpoint))
        print("Eval dir: {}".format(FLAGS.eval_dir))
        train()
        for idx, datadir in enumerate(myDatadirs):
            FLAGS.batches_dir = batches_dir_base  + "_" + datadir
            precision = cifar10_eval.evaluate()
            print("Evaluating {} on {}-trained network (initially trained on {}): {}".format(datadir, dir, FLAGS.prelearned_checkpoint, precision))
            log.write("Evaluating {} on {}-trained network: {}\n".format(datadir, dir, precision))
            log.flush()

def main_justTheOne(argv=None):  # pylint: disable=unused-argument
    FLAGS.run_once = True
    train_dir_base = FLAGS.train_dir
    data_dir_base = FLAGS.data_dir
    batches_dir_base = FLAGS.batches_dir
    eval_dir_base = FLAGS.eval_dir
    checkpoint_dir_base = FLAGS.checkpoint_dir

    #myDatadirs = []
    #myDatadirs = image2vid.generateDatasets('', src_dir, FLAGS.data_dir, '', x264, '', saveFrames, quants)
    #datasetNames = ["yuv", "y_quv", "y_squv", "interlaced"]
    myDatadirs = [FLAGS.single_dir,]
    logfile = os.path.join(FLAGS.mylog_dir, "log_results.txt")
    log = open(logfile, 'w')
    log.write("Here are the results \n")

    saveFrames = (0, 2, 3, 6)
    saveFrames = (0,)
    quants = (10, 25, 37, 41, 46, 50)
    #quants = (10, 25, 37, 50)
    #myDatadirs = ["yuv", "nn", "y_quv", "y_squv", "interlaced"]
    myDatadirs = ["yuv", "y_quv", "y_squv", "interlaced"]
    myDatadirs = ["yuv", "all"]
    for quant in quants:
        for idx, frame in enumerate(saveFrames):
            name = "q{}_f{}".format(quant, saveFrames[idx])
            myDatadirs.append(name)

    print(myDatadirs)
    logfile = os.path.join(FLAGS.mylog_dir, "log.txt")
    log = open(logfile, 'w')
    log.write("Here are the results \n")

    print("For testing: These are my datadirs: {}".format(myDatadirs))

    myname = FLAGS.batches_dir + "_" + FLAGS.single_dir
    name = os.path.join(FLAGS.data_dir, myname)

    FLAGS.train_dir = train_dir_base  + "/train_" + FLAGS.single_dir
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    FLAGS.batches_dir = batches_dir_base  + "_" + FLAGS.single_dir
    FLAGS.checkpoint_dir = FLAGS.train_dir
    print("The single_dir: {}".format(FLAGS.single_dir))
    print("Train dir: {}".format(FLAGS.train_dir))
    print("Data dir: {}".format(FLAGS.data_dir))
    print("Batches dir: {}".format(FLAGS.batches_dir))
    print("Checkpoint dir: {}".format(FLAGS.checkpoint_dir))
    print("Prelearned checkpoint file: {}".format(FLAGS.prelearned_checkpoint))
    print("Eval dir: {}".format(FLAGS.eval_dir))
    train()
    for idx, datadir in enumerate(myDatadirs):
        FLAGS.batches_dir = batches_dir_base  + "_" + datadir
        precision = cifar10_eval.evaluate()
        print("Evaluating {} on {}-trained network): {}".format(datadir, FLAGS.single_dir, FLAGS.prelearned_checkpoint, precision))
        log.write("Evaluating {} on {}-trained network: {}\n".format(datadir, FLAGS.single_dir, precision))
        log.flush()

def main(argv=None):
    main_justTheOne(argv)
    #main_testall(argv)


if __name__ == '__main__':
  tf.app.run()
