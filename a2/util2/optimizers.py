
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

import numpy as np


class ClippedGradientDescentOptimizer(optimizer.Optimizer):
  """Optimizer that implements the clipped gradient descent algorithm."""

  def __init__(self, learning_rate, clip_norm=np.inf, use_locking=False,
               name="ClippedGradientDescent"):
    """Construct a new clipped gradient descent optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      clip_norm: A Tensor or a floating point value.  The bound to impose
        on the gradient norm for use in "tf.clip_by_global_norm".
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".
    """
    super(ClippedGradientDescentOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._clip_norm = clip_norm
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
    self._clip_norm_tensor = ops.convert_to_tensor(self._clip_norm,
                                                   name="clip_norm")

  def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=1, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None):
    grads_and_vars = self.compute_gradients(
        loss, var_list=var_list, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)
    grads, var_list = zip(*grads_and_vars)
    clipped, _ = clip_ops.clip_by_global_norm(grads, self._clip_norm_tensor)
    clipped_grads_and_vars = zip(clipped, var_list)
    return self.apply_gradients(
        clipped_grads_and_vars, global_step=global_step, name=name)

  def _apply_dense(self, grad, var):
    return training_ops.apply_gradient_descent(
        var,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, handle):
    return resource_variable_ops.assign_add_variable_op(
        handle, -grad * self._learning_rate)

  def _resource_apply_sparse(self, grad, handle, indices):
    return resource_variable_ops.resource_scatter_add(
        handle, indices, -grad * self._learning_rate)

  def _apply_sparse(self, grad, var):
    delta = ops.IndexedSlices(
        grad.values *
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.indices, grad.dense_shape)
    return var.scatter_sub(delta, use_locking=self._use_locking)

