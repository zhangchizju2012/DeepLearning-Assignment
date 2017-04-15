
import numpy as np
import tensorflow as tf

def expected_cost(z_hat, y, tau=1.0): # y is indicator
	p_hat = tf.nn.softmax(z_hat/tau)
	return 1.0 - tf.reduce_sum(tf.mul(p_hat, y), axis=1)

def sparse_expected_cost(z_hat, y, tau=1.0): # y is index
  n = z_hat.get_shape().as_list()[1]
  I = np.identity(n, dtype=np.float32)
  y_mat = tf.nn.embedding_lookup(I, y)
  p_hat = tf.nn.softmax(z_hat/tau)
  return 1.0 - tf.reduce_sum(tf.mul(p_hat, y_mat), axis=1)

def kl_divergence_rl(z_hat, y, tau=1.0): # y is indicator
	pot_z_hat = tau*tf.reduce_logsumexp(z_hat/tau, 1)
	pot_y = tau*tf.reduce_logsumexp(y/tau, 1)
	p_hat = tf.nn.softmax(z_hat/tau)
	return pot_y - pot_z_hat - tf.reduce_sum(tf.mul(p_hat, y - z_hat), axis=1)

def sparse_kl_divergence_rl(z_hat, y, tau=1.0): # y is index
  n = z_hat.get_shape().as_list()[1]
  I = np.identity(n, dtype=np.float32)
  y_mat = tf.nn.embedding_lookup(I, y)
  pot_z_hat = tau*tf.reduce_logsumexp(z_hat/tau, 1)
  pot_y = tau*tf.reduce_logsumexp(y_mat/tau, 1)
  p_hat = tf.nn.softmax(z_hat/tau)
  return (pot_y - pot_z_hat - 
          tf.reduce_sum(tf.mul(p_hat, y_mat - z_hat), axis=1))

def kl_divergence_ml(z_hat, y, tau=1.0): # y is indicator
	pot_z_hat = tau*tf.reduce_logsumexp(z_hat/tau, 1)
	pot_y = tau*tf.reduce_logsumexp(y/tau, 1)
	p = tf.nn.softmax(y/tau)
	return pot_z_hat - pot_y - tf.reduce_sum(tf.mul(p, z_hat - y), axis=1)

def sparse_kl_divergence_ml(z_hat, y, tau=1.0): # y is index
  n = z_hat.get_shape().as_list()[1]
  I = np.identity(n, dtype=np.float32)
  y_mat = tf.nn.embedding_lookup(I, y)
  pot_z_hat = tau*tf.reduce_logsumexp(z_hat/tau, 1)
  pot_y = tau*tf.reduce_logsumexp(y_mat/tau, 1)
  p = tf.nn.softmax(y_mat/tau)
  return pot_z_hat - pot_y - tf.reduce_sum(tf.mul(p, z_hat - y_mat), axis=1)

