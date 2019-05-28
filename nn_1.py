import numpy as np
np.random.seed(11)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


train_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_label = np.array([(1, 0), (1, 0), (0, 1), (0, 1)])
test_X = []
test_label = []

n_hidden_units_1 = 3
n_output_units = 2


def fwd_pass(X, y):

    W_1 = np.random.random((X.shape[-1], n_hidden_units_1))
    b1 = np.zeros((n_hidden_units_1))
    W_2 = np.random.random((n_hidden_units_1, n_output_units))
    b2 = np.zeros((n_output_units))

    for i, x in enumerate(X):

        res_1 = np.dot(W_1.T, x) + b1
        res_1 = sigmoid(res_1)
        output = np.dot(W_2.T, res_1) + b2
        output = sigmoid(output)

        bwd_pass(output, y[i], res_1, x)


def bwd_pass(output, y, res_1, ip_1):

    grad_pre_output = -(y - output)
    grad_W_2 = np.dot(np.expand_dims(grad_pre_output, axis=1), np.expand_dims(res_1, axis=0))
    grad_b2 = grad_pre_output

    print(np.dot(grad_W_2, np.expand_dims(ip_1, axis=0).T))
    # grad_W_1 = np.dot
    # grad_b1 =


if __name__ == '__main__':
    fwd_pass(train_X, train_label)
