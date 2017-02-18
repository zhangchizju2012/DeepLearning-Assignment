
"""Simple framework for capturing various measurements recorded
   during an experiment"""


import tensorflow as tf
import numpy as np
import time


class measure():

  def __init__(self, label, axes=[0.0, np.inf, 0.0, np.inf]):
    self.label = label
    self.axes = axes

  def update(self):
    None

  def reset(self):
    None

  def printout(self, sess, echo):
    val = self.eval(sess)
    if echo:
      print "%s %g"%(self.label, val),
    return val


class meas_loss(measure):

  def __init__(self, x, y, data, loss, label,
               batch=100, axes=[0.0, np.inf, 0.0, np.inf]):
    self.x = x
    self.y = y
    self.data = data
    self.loss = loss
    self.label = label
    self.batch = batch
    self.axes = axes

  def eval(self, sess):
    loss_total = 0.0
    X, Y = self.data
    end = X.shape[0]
    cur = 0
    while cur < end:
      inds = range(cur, min(cur + self.batch, end))
      loss_total += sess.run(self.loss, feed_dict={self.x: X[inds],
                                                   self.y: Y[inds]})
      cur += self.batch
    return loss_total


class meas_time(measure):

  def __init__(self, label, axes=[0.0, np.inf, 0.0, np.inf]):
    self.label = label
    self.axes = axes
    self.start_time = None

  def update(self):
    if self.start_time == None:
      self.start_time = time.time()

  def reset(self):
    self.start_time = None

  def eval(self, sess):
    return 0 if self.start_time == None else time.time() - self.start_time


class meas_iter(measure):

  def __init__(self, gap, label, axes=[0.0, np.inf, 0.0, np.inf]):
    self.gap = gap
    self.label = label
    self.axes = axes
    self.iter = None

  def update(self):
    if self.iter == None:
      self.iter = 0
    else:
      self.iter += self.gap

  def reset(self):
    self.iter = None

  def eval(self, sess):
    return self.iter


# general

def reset(meas_list):
  for meas in meas_list:
    meas.reset()

def update(meas_list):
  for meas in meas_list:
    meas.update()

def printout(label, meas_list, sess, echo):
  results = np.zeros(len(meas_list))
  i = 0
  if echo:
    print label,
  for meas in meas_list:
    results[i] = meas.printout(sess, echo)
    i += 1
  if echo:
    print
  return results

