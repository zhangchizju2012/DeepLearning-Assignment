
import numpy as np
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops


class SimpleRNNCell(rnn_cell.RNNCell):

  def __init__(self, hidden_size, input_size=None, activation=nn_ops.relu):
    self._hidden_size = hidden_size
    self._activation = activation

  @property
  def state_size(self):
    return self._hidden_size

  @property
  def output_size(self):
    return self._hidden_size

  def __call__(self, inputs, state, scope=None):
    n = inputs.get_shape().as_list()[1]
    h = self._hidden_size
    with variable_scope.variable_scope(scope or type(self).__name__):
      W = variable_scope.get_variable(
          "W", [n, h],
          initializer=init_ops.random_normal_initializer(
              stddev=1.0/np.sqrt(n + h + 1)))
      U = variable_scope.get_variable(
          "U", [h, h],
          initializer=init_ops.random_normal_initializer(
              stddev=1.0/np.sqrt(n + h + 1)))
      b = variable_scope.get_variable(
          "b", [1, h],
          initializer=init_ops.constant_initializer(1.0/np.sqrt(n + h + 1)))
      next_state = self._activation(
          math_ops.matmul(inputs, W) + math_ops.matmul(state, U) + b)
    return next_state, next_state


class SimpleLSTMCell(rnn_cell.RNNCell):

  def __init__(self, hidden_size=1, forget_bias=1.0, state_is_tuple=True,
               activation=math_ops.tanh):
    self._hidden_size = hidden_size
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation

  @property
  def state_size(self):
    return (self._hidden_size, self._hidden_size if self._state_is_tuple else
            2*self._hidden_size)

  @property
  def output_size(self):
    return self._hidden_size

  def __call__(self, inputs, state, scope=None):
    n = inputs.get_shape().as_list()[1]
    h = self._hidden_size
    with variable_scope.variable_scope(scope or type(self).__name__):
      W = variable_scope.get_variable(
          "W", [n, 4*h],
          initializer=init_ops.random_normal_initializer(
              stddev=1.0/np.sqrt(n + 1)))
      U = variable_scope.get_variable(
          "U", [h, 4*h],
          initializer=init_ops.random_normal_initializer(
              stddev=1.0/np.sqrt(h + 1)))
      b = variable_scope.get_variable(
          "b", [1, 4*h],
          initializer=init_ops.random_normal_initializer(
              stddev=1.0/np.sqrt(n + h + 1)))
    state_c, state_h = (state if self._state_is_tuple else
                        array_ops.split(1, 2, state))
    linear_cat = math_ops.matmul(inputs, W) + math_ops.matmul(state_h, U) + b
    # i=input, u=update, f=forget, o=output
    i, u, f, o = array_ops.split(1, 4, linear_cat)
    new_c = (math_ops.sigmoid(f + self._forget_bias)*state_c +
             math_ops.sigmoid(i)*self._activation(u))
    new_h = math_ops.sigmoid(o)*self._activation(new_c)
    next_state = (new_c, new_h if self._state_is_tuple else
                  array_ops.concat(1, [new_c, new_h]))
    return new_h, next_state

