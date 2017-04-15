
  # set up paths
import os, sys
homepath = os.getcwd() + "/../.."
datapath = homepath + "/data2/"
sys.path.append(homepath)

import numpy as np
import tensorflow as tf
import util2
from util2 import measurement as meas

# experimental configuration
BATCH = 1 # minibatch size
STEPS = 15 # rollout length

# plot bounds
YMAX_TRAIN = np.inf

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

def numberToString(number):
  string = ''
  for item in number:
    string += id2char[item]
  return string

def stringToNumber(string):
  number = []
  for item in string:
    if item in char2id:
      number.append(char2id[item])
    else:
      number.append(0)
  return number

def generate(methods,variables_file,seed_string,length):
  """Run an experiment over all methods"""
  saver = tf.train.Saver()
  tf.get_default_graph().finalize()
  with tf.Session() as sess:
    for method in methods:
      meas.reset(method.meas)
      for var in tf.global_variables():
        sess.run(var.initializer)
        saver.restore(sess,variables_file)        
      meas.update(method.meas)
      for mea in method.meas:
        cur_state = sess.run(mea.zero_state)
        X = seed_string
        Y = " " * STEPS
        mea.ydata[:] = stringToNumber(Y)
        start = len(X) - STEPS
        while len(X) <= length:
          #print start
          mea.xdata[:] = stringToNumber(X[start:start+STEPS])
          mea.feed_dict[mea.init_state] = cur_state
          cur_loss, cur_state = sess.run((mea.loss, mea.final_state),
                                         feed_dict=mea.feed_dict)
          probability = []
          for item in cur_loss[-1]:
            probability.append(item/sum(cur_loss[-1]))
          temp = np.random.choice(np.arange(50),p=probability)
          X += numberToString([temp])
          start += 1
        print X
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
    self.y_hat = y_hat
      # loss
    self.train_loss = tf.reduce_mean(loss_fun(z_hat, self.y))
    self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=name)
    self.misclass_err = tf.reduce_mean(
        tf.cast(tf.not_equal(tf.cast(tf.argmax(y_hat, 1), tf.int32), self.y),
                tf.float32))
    self.predictMine = tf.argmax(y_hat, 1)


# wrapper: combines model + optimizer into "method" to run with util2.experiment

def methoddef(name, color, model, optimizer,
              xdata, ydata, data_train, data_valid, data_test):
  """method = model + optimizer"""
  method = util2.experiment.method_rnn(
      name, color, model, optimizer, data_train, xdata, ydata)
  method.meas = [meas.meas_rnnloss(model.x, model.y,
                                   model.zero_state, model.init_state,
                                   model.final_state,
                                   data_train, model.y_hat, "y_hat",
                                   BATCH, STEPS,
                                   axes=[0.0, np.inf, 0.0, YMAX_TRAIN])]
  return method


# define methods
def main():
    methods = []
    
    loss_kl_base2 = (lambda zhat, y:
        tf.nn.sparse_softmax_cross_entropy_with_logits(zhat, y) / np.log(2))
    
    '''
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
    '''
    name = "LSTM_base"
    color = "Red"
      # model
    hidden = 200
    depth = 2
    cell_type = util2.rnn_simple.SimpleLSTMCell
    gate_fun = tf.nn.tanh
    loss_fun = loss_kl_base2
    dimensions = hidden, depth, vocab
    model = model_rnn(name, cell_type, dimensions, gate_fun, loss_fun)
      # optimizer
    init_step = 10.
    step_size_fun = (lambda global_step:
                        util2.step_sizers.exp_stair_step(
                           init_step, global_step, steps_per_epoch=epoch, delay=4))
    optim_fun = (lambda stepsize:
                    util2.optimizers.ClippedGradientDescentOptimizer(
                        stepsize, clip_norm=0.2))
    optimizer = util2.experiment.optimizer(optim_fun, step_size_fun)
      # method
    method = methoddef(name, color, model, optimizer, xdata, ydata,
                       data_train, data_valid, data_test)
    methods.append(method)
    
    # run experiment
    
    #methods_use = [methods[1]]
    methods_use = methods
    
    variables_file = '/Users/zhangchi/Desktop/DeepLearning-Assignment/a2/experiments2/3_losses/out/3b_loss/3b_loss'
    #variables_file = '/Users/zhangchi/Desktop/DeepLearning-Assignment/a2/experiments2/2_gates/out/2c_gate/2c_gate'
    #seed_string = "after the next crash"
    #seed_string = "the president denied"
    #seed_string = "money, its a hit"
    seed_string = "the financial sector"
    length = 256
    generate(methods_use,variables_file,seed_string,length)
    
main()

