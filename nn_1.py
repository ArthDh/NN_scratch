import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


X = np.array([[0, 0, 1, 1], [1, 1, 1, 1]])

n_hidden_units_1 = 3

n_output_units = 1

for x in X:
    W_1 = np.ones((X.shape[-1], n_hidden_units_1))
    b1 = np.ones((n_hidden_units_1))
    res_1 = np.dot(W_1.T, x) + b1
    res_1 = sigmoid(res_1)
    W_2 = np.ones((n_hidden_units_1, n_output_units))
    b2 = np.ones((n_output_units))
    output = np.dot(W_2.T, res_1) + b2
    output = sigmoid(output)

    print(output)
