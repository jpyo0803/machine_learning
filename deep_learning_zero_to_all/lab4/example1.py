import numpy as np
import tensorflow as tf

tf.random.set_seed(0)

# data and label
X1 = [73., 93., 89., 96., 73.]
X2 = [80., 88., 91., 98., 66.]
X3 = [75., 93., 90., 100., 70.]
Y = [152., 185., 180., 196., 142.]

# random weights
w1 = tf.Variable(tf.random.normal([1]))
w2 = tf.Variable(tf.random.normal([1]))
w3 = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.random.normal([1]))

learning_rate = 0.000001

for i in range(1000 + 1):
    with tf.GradientTape() as tape:
        hypothesis = w1 * X1 + w2 * X2 + w3 * X3 + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

    w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])

    w1.assign_sub(learning_rate * w1_grad)
    w2.assign_sub(learning_rate * w2_grad)
    w3.assign_sub(learning_rate * w3_grad)  
    b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
        print("{:5} | {:12.4f}".format(i, cost.numpy()))
