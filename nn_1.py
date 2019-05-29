import numpy as np
np.random.seed(11)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


train_X = np.array([[0,0],[0,1],[1,0],[1,1]])
train_label = np.array([[1],[1],[0],[0]])
test_X = []
test_label = []


n_hidden_units_1 = 3
n_output_units = 1


def train(X, y, lr=0.01, epochs=10):

    W_1 = np.random.random((X.shape[-1], n_hidden_units_1))
    b1 = np.zeros((n_hidden_units_1))
    W_2 = np.random.random((n_hidden_units_1, n_output_units))
    b2 = np.zeros((n_output_units))
    lamda = 0.7

    for epoch in range(epochs):
        for i, x in enumerate(X):

            # FWD Pass
            res_1 = np.dot(W_1.T, x) + b1
            res_1_activation = sigmoid(res_1)            

            output = np.dot(W_2.T, res_1_activation) + b2
            output_activation = sigmoid(output)
    

            # BWD Pass
            grad_pre_output = -(y[i] - output_activation) 
            grad_W_2 = np.dot(np.expand_dims(res_1_activation, axis=1),np.expand_dims(grad_pre_output, 1).T)
            grad_b2 = np.expand_dims(grad_pre_output, 1)

            grad_post_1 = np.dot(W_2, np.expand_dims(grad_pre_output, axis=1))
            grad_pre_1 =np.multiply(grad_post_1.T, d_sigmoid(res_1))
            grad_W_1 = np.dot(np.expand_dims(x, axis=1), grad_pre_1)
            grad_b1 = grad_pre_1.T

            W_1 = np.add(W_1, lr * ((grad_W_1) - lamda * 2 * W_1))
            W_2 = np.add(W_2, lr * ((grad_W_2) -  lamda * 2 * W_2))
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


    # print(model[2])
    # print(model[3])
    predict([1, 1], model)
