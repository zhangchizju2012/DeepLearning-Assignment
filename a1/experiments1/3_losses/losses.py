'''
import tensorflow as tf

def f_tau(x, tau):
	num = tf.exp(tf.scalar_mul(1/tau,x))
	den = 1 / tf.reduce_sum(num,axis=1,keep_dims=True)
	return tf.multiply(num, den)

def expected_cost(z_hat, y, tau=1.0):
	c = 1 - y
	return tf.multiply(f_tau(z_hat,tau),c)

def F(x):
	num = tf.reduce_sum(tf.exp(x),axis=1,keep_dims=True)
	return tf.log(num)

def F_tau(x , tau):
	return tau * F(x / tau)

def F_star(x):
    temp = tf.matmul(x,tf.log(tf.transpose(x)))
    return tf.expand_dims(tf.diag_part(temp), 1)

def F_tau_star(x, tau):
	return tau * F_star(x)

def kl_divergence_rl(z_hat, y, tau=0.5):
	return F_tau(y, tau) - tf.reduce_sum(tf.multiply(y,f_tau(z_hat,tau)),axis=1,keep_dims=True) + F_tau_star(f_tau(z_hat,tau), tau)

def kl_divergence_ml(z_hat, y, tau=1):
	return F_tau(z_hat, tau) - tf.reduce_sum(tf.multiply(z_hat,f_tau(y,tau)),axis=1,keep_dims=True) + F_tau_star(f_tau(y,tau), tau)
'''
import numpy as np
import tensorflow as tf

def f_tau(x, tau):
    ex = tf.exp(tf.scalar_mul(1.0 / tau, x))
    den = tf.reduce_sum(ex, axis=1, keep_dims=True)
    den = tf.tile(den, [1, tf.shape(x)[1]])
    return tf.mul(1.0 / den, ex)

def expected_cost(z_hat, y, tau=1.0):
    c = tf.add(1.0, -y)
    return tf.reduce_sum(tf.mul(f_tau(z_hat, tau), c), 1)

def F_tau(x, tau):
    ex = tf.exp(tf.scalar_mul(1.0 / tau, x))
    return tf.scalar_mul(tau, tf.log(tf.reduce_sum(ex, axis=1)))

def F_star_tau(x, tau):
    logx = tf.log(x)
    return tf.scalar_mul(tau, tf.reduce_sum(tf.mul(x, logx), axis=1))

def kl_divergence_rl(z_hat, y, tau=1.0):
    f_tau_z_hat = f_tau(z_hat, tau)
    return F_tau(y, tau) - tf.reduce_sum(tf.mul(y, f_tau_z_hat), axis=1) + F_star_tau(f_tau_z_hat, tau)

def kl_divergence_ml(z_hat, y, tau=1.0):
    f_tau_y = f_tau(y, tau)
    return F_tau(z_hat, tau) - tf.reduce_sum(tf.mul(z_hat, f_tau_y), axis=1) + F_star_tau(f_tau_y, tau)
