import tensorflow as tf

def f_tau(x, tau):
	num = tf.exp(tf.scalar_mul(1/tau,x))
	den = 1 / tf.reduce_sum(num,axis=1,keep_dims=True)
	return tf.multiply(num, den)

def expected_cost(z_hat, y, tau=1.0):
	c = 1 - y
	return tf.multiply(f_tau(z_hat,tau),c)
