import numpy as np

X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])


def cost_func(W, X, Y):
    c = 0
    m = len(X)
    for i in range(m):
        c += (W * X[i] - Y[i]) ** 2
    return c / m


for feed_W in np.linspace(-3, 5, num=15):
    current_cost = cost_func(feed_W, X, Y)
    print("{:6.3f} | {:10.5f}".format(feed_W, current_cost))
