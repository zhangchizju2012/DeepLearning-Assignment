
  # modify paths to match your own set-up
homepath = "/Users/zhangchi/Desktop/"
#homepath = "/home/ubuntu/src/tensorflow/dl/"
datapath = homepath + "DeepLearning-Assignment/a1/data1"
srcpath = homepath + "DeepLearning-Assignment/a1"

import sys
sys.path.append(srcpath)

import numpy as np
import tensorflow as tf
import util1
import pickle
import losses
from util1 import measurement as meas


# experimental configuration
REPEATS = 1
MAX_EPOCH = 20
BATCH = 100 # minibatch size
VALID = 5000 # size of validation set
SEED = 66478 # None for random seed
PERMUTE = False # permute pixels
ECHO = True

# save filenames
FILELABEL = "loss"
FILETAG = "3a"

# plot bounds
YMAX_TRAIN = 10000.0
YMAX_VALID = 200.0
YMAX_TEST  = 400.0


epoch = (60000 - VALID)//BATCH


# data

  # read MNIST data: vector input format (used by fully connected model)
data_train_vec, data_valid_vec, data_test_vec = util1.read_mnist.read_data(
    datapath, image_shape=[784], image_range=[0.0, 1.0], one_hot=True,
    num_validation=VALID, permute_pixels=PERMUTE)
  # dimensions
t, n = data_train_vec[0].shape
te, m = data_test_vec[1].shape
  # minibatch holders
xdata_vec = np.zeros([BATCH, n], dtype=np.float32)
ydata = np.zeros([BATCH, m], dtype=np.float32)

  # read in MNIST data: matrix input format (used by convolutional model)
data_train_mat, data_valid_mat, data_test_mat = util1.read_mnist.read_data(
    datapath, image_shape=[28, 28], image_range=[0.0, 1.0], one_hot=True,
    num_validation=VALID, permute_pixels=PERMUTE)
t, n1, n2, d0 = data_train_mat[0].shape # dimensions
xdata_mat = np.zeros([BATCH, n1, n2, d0], dtype=np.float32) # minibatch holder


# model architectures

class model_f_f():
  """Define an f_f model: input -> fully connected -> output"""
  def __init__(self, name, dimensions, gate_fun, loss_fun):
      # placeholders
    dim_in, hidden, dim_out = dimensions
    self.x = tf.placeholder(tf.float32, shape=(None, dim_in)) # input
    self.y = tf.placeholder(tf.float32, shape=(None, dim_out)) # target
      # layer 1: full
    W_1, b_1, z_hat_1, y_hat_1 = util1.layers.fully_connected(
        name, "layer_1", self.x, dim_in, hidden,
        tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_in+1), seed=SEED),
        gate_fun)
      # layer 2: full
    W_2, b_2, z_hat, y_hat = util1.layers.fully_connected(
        name, "layer_2", y_hat_1, hidden, dim_out,
        tf.random_normal_initializer(stddev=1.0/np.sqrt(hidden+1), seed=SEED),
        tf.nn.softmax)
      # loss
    self.train_loss = tf.reduce_sum(loss_fun(z_hat, self.y))
    self.train_vars = [W_1, b_1, W_2, b_2]
    self.misclass_err = tf.reduce_sum(tf.cast(
        tf.not_equal(tf.argmax(y_hat, 1), tf.argmax(self.y, 1)), tf.float32))

class model_cp_cp_f():
  """Define a cp_cp_f model: input -> conv & pool -> conv & pool -> output"""
  def __init__(self, name, dimensions, gate_fun, loss_fun):
      # placeholders
    n1, n2, d0, f1, d1, f2, d2, m = dimensions
    self.x = tf.placeholder(tf.float32, shape=(None, n1, n2, d0)) # input
    self.y = tf.placeholder(tf.float32, shape=(None, m)) # target
      # layer 1: conv
    W_1, b_1, z_hat_1, r_hat_1 = util1.layers.convolution_2d(
        name, "layer_1", self.x, f1, d0, d1,
        tf.random_normal_initializer(stddev=1.0/np.sqrt(f1*f1*d0+1), seed=SEED),
        gate_fun)
      # layer 1.5: pool
    s_hat_1 = tf.nn.max_pool(
        r_hat_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
      # layer 2: conv
    W_2, b_2, z_hat_2, r_hat_2 = util1.layers.convolution_2d(
        name, "layer_2", s_hat_1, f2, d1, d2,
        tf.random_normal_initializer(stddev=1.0/np.sqrt(f2*f2*d1+1), seed=SEED),
        gate_fun)
      # layer 2.5: pool
    s_hat_2 = tf.nn.max_pool(
        r_hat_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    shape_2 = s_hat_2.get_shape().as_list()
    y_hat_2 = tf.reshape(s_hat_2, [-1, shape_2[1]*shape_2[2]*shape_2[3]])
      # layer 3: full
    W_3, b_3, z_hat, y_hat = util1.layers.fully_connected(
        name, "layer_3", y_hat_2, (n1*n2*d2)//16, m,
        tf.random_normal_initializer(stddev=1.0/np.sqrt((n1*n2*d2)//16),
                                     seed=SEED),
        tf.nn.softmax)
      # loss
    self.train_loss = tf.reduce_sum(loss_fun(z_hat, self.y))
    self.train_vars = [W_1, b_1, W_2, b_2, W_3, b_3]
    self.misclass_err = tf.reduce_sum(tf.cast(
        tf.not_equal(tf.argmax(y_hat, 1), tf.argmax(self.y, 1)), tf.float32))


# wrapper: combines model + optimizer into "method" to run with util1.experiment

def methoddef(name, color, model, optimizer,
              xdata, ydata, data_train, data_valid, data_test):
  """method = model + optimizer:
     wrap a model + optimizer into a method for use with util1.experiment"""
  method = util1.experiment.method(
      name, color, model, optimizer, data_train, xdata, ydata)
  method.meas = [meas.meas_iter(epoch, "step"),
                 meas.meas_loss(model.x, model.y, data_train,
                                model.train_loss, "train_loss", batch=BATCH,
                                axes=[0.0, np.inf, 0.0, YMAX_TRAIN]),
                 meas.meas_loss(model.x, model.y, data_valid,
                                model.misclass_err, "valid_err", batch=BATCH,
                                axes=[0.0, np.inf, 0.0, YMAX_VALID]),
                 meas.meas_loss(model.x, model.y, data_test,
                                model.misclass_err, "test_err", batch=BATCH,
                                axes=[0.0, np.inf, 0.0, YMAX_TEST]),
                 meas.meas_time("train_time") ]
  return method

# leaky Relu
def my_gate(z):
  return tf.maximum(0.1*z,z)

# define methods

methods = []

# method a
name = "f_f-relu"
color = "Blue"
hidden = 1024
dimensions = (n, hidden, m)
gate_fun = tf.nn.relu
loss_fun = losses.expected_cost
model = model_f_f(name, dimensions, gate_fun, loss_fun)
optimizer = tf.train.MomentumOptimizer(0.1/BATCH, momentum=0.9)
method = methoddef(name, color, model, optimizer, xdata_vec, ydata,
                   data_train_vec, data_valid_vec, data_test_vec)
methods.append(method)

# run experiment

#methods_use = [methods[3]]
#methods_use = [methods[1]]
methods_use = methods

gap_timer = util1.gap_timing.gaptimer(epoch, MAX_EPOCH)
sampler = util1.wrap_counting.wrapcounter(BATCH, t, seed=SEED)

results = util1.experiment.run_methods_list(
    methods_use, gap_timer, sampler, REPEATS, ECHO)

means = util1.experiment.summarize(results) # updates, methods, measures
util1.experiment.print_results(methods_use, means, sys.stdout, FILETAG)
util1.experiment.print_results(methods_use, means, FILELABEL, FILETAG)
pickle.dump( [means, FILELABEL, FILETAG], open( "saveA.p", "wb" ) )
util1.experiment.plot_results(methods_use, means, FILELABEL, FILETAG)

