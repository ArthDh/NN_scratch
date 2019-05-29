import numpy as np
np.random.seed(11)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


train_X = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0]])
train_label = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]])
test_X = []
test_label = []

n_hidden_units_1 = 3
n_output_units = 2


def train(X, y, lr=0.01, epochs=1):

    W_1 = np.random.random((X.shape[-1], n_hidden_units_1))
    b1 = np.zeros((n_hidden_units_1))
    W_2 = np.random.random((n_hidden_units_1, n_output_units))
    b2 = np.zeros((n_output_units))

    for epoch in range(epochs):
        for i, x in enumerate(X):
            res_1 = np.dot(W_1.T, x) + b1
            res_1_activation = sigmoid(res_1)

            output = np.dot(W_2.T, res_1_activation) + b2
            output_activation = sigmoid(output)
            # bwd_pass(output_activation, output, y[i], res_1_activation, res_1, x)

            grad_pre_output = -(y[i] - output_activation)
            grad_W_2 = np.dot(np.expand_dims(grad_pre_output, 1), np.expand_dims(res_1_activation, axis=1).T)
            grad_b2 = np.expand_dims(grad_pre_output, 1)

            grad_post_1 = np.dot(np.expand_dims(grad_pre_output, axis=0), W_2.T)
            grad_pre_1 = grad_post_1 * d_sigmoid(res_1)

            grad_W_1 = np.dot(np.expand_dims(x, axis=1), grad_pre_1)
            grad_b1 = grad_pre_1.T

            # print(grad_W_1.shape, grad_W_2.shape, grad_b1.shape, grad_b2.shape)

            W_1 = np.add(W_1, lr * ((grad_W_1) - 2 * W_1))

            W_2 = np.add(W_2, lr * ((grad_W_2.T) - 2 * W_2))
            b1 = np.add(b1, np.reshape(grad_b1, (grad_b1.shape[0],)))
            b2 = np.add(b2, np.reshape(grad_b2, (grad_b2.shape[0],)))

    return (W_1, W_2, b1, b2)


def predict(x, model):

    (W_1, W_2, b1, b2) = model

    res_1 = np.dot(W_1.T, x) + b1
    res_1_activation = sigmoid(res_1)
    output = np.dot(W_2.T, res_1_activation) + b2
    output_activation = sigmoid(output)

    print(output_activation)


if __name__ == '__main__':
    model = train(train_X, train_label)
    predict([0, 0, 0], model)
