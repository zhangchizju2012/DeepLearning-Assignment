
"""Functions for defining neural network layers in tensorflow"""

from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

def fully_connected(name, label, var_in, dim_in, dim_out,
                    initializer, transfer, reuse=False):
  """Standard fully connected layer"""
  with variable_scope.variable_scope(name, reuse=reuse):
    with variable_scope.variable_scope(label, reuse=reuse):
      if reuse:
        W = variable_scope.get_variable("W", [dim_in, dim_out])
        b = variable_scope.get_variable("b", [dim_out])
      else: # new
        W = variable_scope.get_variable("W", [dim_in, dim_out],
                                        initializer=initializer)
        b = variable_scope.get_variable("b", [dim_out],
                                        initializer=initializer)
  z_hat = math_ops.matmul(var_in, W)
  z_hat = nn_ops.bias_add(z_hat, b)
  y_hat = transfer(z_hat)
  return W, b, z_hat, y_hat


def convolution_2d(name, label, var_in, f, dim_in, dim_out,
                   initializer, transfer, reuse=False):
  """Standard convolutional layer"""
  with variable_scope.variable_scope(name, reuse=reuse):
    with variable_scope.variable_scope(label, reuse=reuse):
      if reuse:
        W = variable_scope.get_variable("W", [f, f, dim_in, dim_out])
        b = variable_scope.get_variable("b", [dim_out])
      else: # new
        W = variable_scope.get_variable("W", [f, f, dim_in, dim_out],
                                        initializer=initializer)
        b = variable_scope.get_variable("b", [dim_out],
                                        initializer=initializer)
  z_hat = nn_ops.conv2d(var_in, W, strides=[1,1,1,1], padding="SAME")
  z_hat = nn_ops.bias_add(z_hat, b)
  y_hat = transfer(z_hat)
  return W, b, z_hat, y_hat

