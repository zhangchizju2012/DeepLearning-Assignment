
"""Simple framework for running experiments on multiple methods:
   method = model + optimizer"""

import tensorflow as tf
import numpy as np
import time
import measurement as meas
import matplotlib.pyplot as plt
import sys


class method():
  """Define the method structure used to run an experiment"""
  def __init__(self, label, color, model, optimizer, data, xdata, ydata):
    self.label = label
    self.color = color
    self.scope_name = label
    self.train_step = optimizer.minimize(model.train_loss,
                                         var_list=model.train_vars)
    self.data = data
    self.xdata = xdata
    self.ydata = ydata
    self.feed_dict = {model.x: self.xdata, model.y: self.ydata}
    self.meas = None


def run_methods_list(methods, gap_timer, sampler, repeats, echo=False):
  """Run an experiment over all methods"""
  tf.get_default_graph().finalize()
  num_meas = 0
  for method in methods:
    if len(method.meas) > num_meas:
      num_meas = len(method.meas)
  results = np.zeros([repeats, gap_timer.maxupdate+1, len(methods), num_meas])
  with tf.Session() as sess:
    for r in range(repeats):
      print "repeat ", r
      for (m, method) in enumerate(methods):
        meas.reset(method.meas)
        gap_timer.reset()
        u = 0
        for var in tf.global_variables():
          sess.run(var.initializer)
        meas.update(method.meas)
        results[r,u,m,:] = meas.printout(method.label, method.meas, sess, echo)
        while gap_timer.alive():
          inds = sampler.next_inds()
          method.xdata[:] = method.data[0][inds]
          method.ydata[:] = method.data[1][inds]
          sess.run(method.train_step, feed_dict=method.feed_dict)
          if gap_timer.update():
            u += 1
            meas.update(method.meas)
            results[r,u,m,:] = meas.printout(method.label, method.meas,
                                             sess, echo)
        if not echo:
          meas.printout(method.label, method.meas, sess, True)
  print
  return results


def summarize(results):
  return np.mean(results, axis=0) # updates, methods, measures


def print_results(methods, mean_results, filename, label):
  if filename == sys.stdout:
    file = sys.stdout
  else:
    file = open(label+"_"+filename+".txt", "w")
  updates = mean_results.shape[0]
  m = -1
  for method in methods:
    m += 1
    for u in range(updates):
      file.write(method.label+"\t")
      s = -1
      for meas in method.meas:
        s += 1
        file.write("%g\t"%(mean_results[u, m, s]))
      file.write("\n")
  if file != sys.stdout:
    file.close()


def plot_results(methods, mean_results, experiment_name, label):
  num_update, _, num_meas = mean_results.shape
  for meas in range(num_meas):
    xmin = max(methods[0].meas[meas].axes[0], 0.0)
    xmax = min(methods[0].meas[meas].axes[1], num_update-1)
    ymin = max(methods[0].meas[meas].axes[2], 0.0)
    ymax = min(methods[0].meas[meas].axes[3], np.amax(mean_results[:,:,meas]))
    xlabel = "update batches"
    ylabel = methods[0].meas[meas].label
    #print methods[0].meas[meas].axes[0]
    #print methods[0].meas[meas].axes[1]
    #print methods[0].meas[meas].axes[2]
    #print methods[0].meas[meas].axes[3]
    #print xmin, xmax, ymin, ymax
    f, ax = plt.subplots()
    ax.set_autoscale_on(False)
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    m = -1
    for method in methods:
      m += 1
      ax.plot(range(num_update), mean_results[:, m, meas],
              color=method.color, label=method.label)
    fig_name = experiment_name + "_" + methods[0].meas[meas].label
    fig_name_ = experiment_name + "_" + methods[0].meas[meas].label
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(fig_name)
    plt.legend()
    plt.savefig(label + "_" + fig_name)
  plt.show()
  plt.close()

