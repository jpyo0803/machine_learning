import tensorflow as tf
import numpy as np

X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])


def cost_func(W, X, Y):
    hypothesis = X * W
    return tf.reduce_mean(tf.square(hypothesis - Y))


W_values = np.linspace(-3, 5, num=15)
cost_values = []

for feed_W in W_values:
    current_cost = cost_func(feed_W, X, Y)
    cost_values.append(current_cost)
    print("{:6.3f} | {:10.5f}".format(feed_W, current_cost))
