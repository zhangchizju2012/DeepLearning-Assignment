
  # set up paths
import os, sys
homepath = os.getcwd() + "/../.."
datapath = homepath + "/data2/"
sys.path.append(homepath)

import numpy as np
import tensorflow as tf
import util2
import pickle
from util2 import measurement as meas

# experimental configuration
REPEATS = 1
MAX_EPOCH = 10
BATCH = 20 # minibatch size
STEPS = 20 # rollout length
ECHO = True

# save filenames
FILELABEL = "gate"
FILETAG = "2c"

# plot bounds
YMAX_TRAIN = np.inf
YMAX_VALID = np.inf
YMAX_TEST  = np.inf


# data

train_file = "ptb.train.txt"
valid_file = "ptb.valid.txt"
test_file  = "ptb.test.txt"
filenames = train_file, valid_file, test_file

data, dicts, vocab = util2.readtext_char.read_data(datapath, filenames)
data_train, data_valid, data_test = data
char2id, id2char = dicts
gap = BATCH*STEPS
epoch = len(data_train[0]) // BATCH // STEPS
#minibatch holders
xdata = np.zeros([gap], dtype=np.int32)
ydata = np.zeros([gap], dtype=np.int32)


# model architecture

class model_rnn():
  def __init__(self, name, cell_type, dimensions, gate_fun, loss_fun):
    hidden, depth, vocab = dimensions
      # placeholders
    self.x = tf.placeholder(tf.int32, shape=(gap)) # inputs
    self.y = tf.placeholder(tf.int32, shape=(gap)) # targets
      # rnn inputs
    I = np.identity(vocab, dtype=np.float32)
    X = tf.reshape(self.x, [BATCH, STEPS])
    inputs = tf.nn.embedding_lookup(I, X)
    inputs_list = tf.unstack(inputs, num=STEPS, axis=1)
      # rnn cells
    cell = cell_type(hidden, activation=gate_fun)
    cell_multi = tf.nn.rnn_cell.MultiRNNCell([cell]*depth)
    self.zero_state = cell_multi.zero_state(BATCH, tf.float32)
    self.init_state = cell_multi.zero_state(BATCH, tf.float32)
      # rnn outputs
    outputs_list, self.final_state = tf.nn.rnn(
        cell_multi, inputs_list, initial_state=self.init_state, scope=name)
    outputs = tf.reshape(tf.concat(1, outputs_list), [-1, hidden])
      # predictor
    W, b, z_hat, y_hat = util2.layers.fully_connected(
        name, "predict", outputs, hidden, vocab,
        tf.random_normal_initializer(stddev=1.0/np.sqrt(hidden+1)),
        tf.nn.softmax)
      # loss
    self.train_loss = tf.reduce_mean(loss_fun(z_hat, self.y))
    self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=name)
    self.misclass_err = tf.reduce_mean(
        tf.cast(tf.not_equal(tf.cast(tf.argmax(y_hat, 1), tf.int32), self.y),
                tf.float32))


# wrapper: combines model + optimizer into "method" to run with util2.experiment

def methoddef(name, color, model, optimizer,
              xdata, ydata, data_train, data_valid, data_test):
  """method = model + optimizer"""
  method = util2.experiment.method_rnn(
      name, color, model, optimizer, data_train, xdata, ydata)
  method.meas = [meas.meas_iter(epoch, "step"),
                 meas.meas_rnnloss(model.x, model.y,
                                   model.zero_state, model.init_state,
                                   model.final_state,
                                   data_train, model.train_loss, "train_loss",
                                   BATCH, STEPS,
                                   axes=[0.0, np.inf, 0.0, YMAX_TRAIN]),
                 meas.meas_rnnloss(model.x, model.y,
                                   model.zero_state, model.init_state,
                                   model.final_state,
                                   data_valid, model.train_loss, "valid_loss",
                                   BATCH, STEPS,
                                   axes=[0.0, np.inf, 0.0, YMAX_VALID]),
                 meas.meas_time("epoch_time") ]
  return method


# define methods

methods = []

loss_kl_base2 = (lambda zhat, y:
    tf.nn.sparse_softmax_cross_entropy_with_logits(zhat, y) / np.log(2))


name = "RNN2_tanh"
color = "Red"
  # model
hidden = 800
depth = 2
cell_type = util2.rnn_simple.SimpleRNNCell
gate_fun = tf.nn.tanh
loss_fun = loss_kl_base2
dimensions = hidden, depth, vocab
model = model_rnn(name, cell_type, dimensions, gate_fun, loss_fun)
  # optimizer
init_step = 0.5
step_size_fun = (lambda global_step:
                    util2.step_sizers.exp_stair_step(
                       init_step, global_step, steps_per_epoch=epoch, delay=4))
optim_fun = (lambda stepsize:
                util2.optimizers.ClippedGradientDescentOptimizer(
                    stepsize, clip_norm=1.0))
optimizer = util2.experiment.optimizer(optim_fun, step_size_fun)
  # method
method = methoddef(name, color, model, optimizer, xdata, ydata,
                   data_train, data_valid, data_test)
methods.append(method)


name = "RNN2_relu"
color = "Blue"
  # model
hidden = 800
depth = 2
cell_type = util2.rnn_simple.SimpleRNNCell
gate_fun = tf.nn.relu
loss_fun = loss_kl_base2
dimensions = hidden, depth, vocab
model = model_rnn(name, cell_type, dimensions, gate_fun, loss_fun)
  # optimizer
init_step = 0.5
step_size_fun = (lambda global_step:
                    util2.step_sizers.exp_stair_step(
                       init_step, global_step, steps_per_epoch=epoch, delay=4))
optim_fun = (lambda stepsize:
                util2.optimizers.ClippedGradientDescentOptimizer(
                    stepsize, clip_norm=1.0))
optimizer = util2.experiment.optimizer(optim_fun, step_size_fun)
  # method
method = methoddef(name, color, model, optimizer, xdata, ydata,
                   data_train, data_valid, data_test)
methods.append(method)


# run experiment

methods_use = methods

out_name = FILETAG + "_" + FILELABEL
out_dir = "out/" + out_name
if not os.path.exists(out_dir):
  os.makedirs(out_dir)
save_file = out_dir + "/" + out_name

gap_timer = util2.gap_timing.gaptimer(epoch, MAX_EPOCH)
sampler = util2.wrap_counting.runcounter(BATCH, STEPS, len(data_train[0]))

results = util2.experiment.run_rnn_methods_list(
    methods_use, gap_timer, sampler, REPEATS, save_file, ECHO)

means = util2.experiment.summarize(results) # updates, methods, measures
util2.experiment.print_results(methods_use, means, sys.stdout, None, None)
util2.experiment.print_results(methods_use, means, out_dir, FILELABEL, FILETAG)
pickle.dump( [means, out_dir, FILELABEL, FILETAG], open( "saveC.p", "wb" ) )
util2.experiment.plot_results(methods_use, means, out_dir, FILELABEL, FILETAG)
