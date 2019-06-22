import numpy as np
from preprocess import *
np.random.seed(13)

n_hidden_units_1 = 7
n_output_units = 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def train(X, y, lr=0.1, epochs=10000):

    W_1 = np.random.random((X.shape[-1], n_hidden_units_1))
    b1 = np.zeros((n_hidden_units_1))
    W_2 = np.random.random((n_hidden_units_1, n_output_units))
    b2 = np.zeros((n_output_units))
    lamda = 0.0

    for epoch in range(epochs):
        for i, x in enumerate(X):
            # FWD Pass
            res_1 = np.dot(W_1.T, x) + b1
            res_1_activation = sigmoid(res_1)
            # print(res_1_activation)
            output = np.dot(W_2.T, res_1_activation) + b2
            output_activation = sigmoid(output)
            # print(output_activation)

            # BWD Pass
            grad_pre_output = -(y[i] - output_activation)
            grad_W_2 = np.dot(np.expand_dims(res_1_activation, axis=1), np.expand_dims(grad_pre_output, 1).T)
            grad_b2 = np.expand_dims(grad_pre_output, 1)

            grad_post_1 = np.dot(W_2, np.expand_dims(grad_pre_output, axis=1))
            grad_pre_1 = np.multiply(grad_post_1.T, d_sigmoid(res_1_activation))
            grad_W_1 = np.dot(np.expand_dims(x, axis=1), grad_pre_1)
            grad_b1 = grad_pre_1.T

            W_1 = np.subtract(W_1, lr * ((grad_W_1) + lamda * np.array([ x/abs(x) for x in W_1])))
            W_2 = np.subtract(W_2, lr * ((grad_W_2) + lamda * np.array([x/abs(x) for x in W_2])))
            b1 = np.subtract(b1, np.reshape(grad_b1, (grad_b1.shape[0],)))
            b2 = np.subtract(b2, np.reshape(grad_b2, (grad_b2.shape[0],)))

    return (W_1, W_2, b1, b2)

def calc_max_val(model):
    return np.max([np.max(model[0]), np.max(model[1])])


def predict(x, model):

    (W_1, W_2, b1, b2) = model

    print(W_1)
    print(W_2)

    res_1 = np.dot(W_1.T, x) + b1
    res_1_activation = sigmoid(res_1)
    output = np.dot(W_2.T, res_1_activation) + b2
    output_activation = sigmoid(output)

    print(output_activation)


if __name__ == '__main__':

    train_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    train_label = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    test_X = []
    test_label = []

    # train_data_path = '/Users/arth/Desktop/Programming/AI_Rutgers/Final_project/data/facedata/facedatatrain'
    # train_labels_path = '/Users/arth/Desktop/Programming/AI_Rutgers/Final_project/data/facedata/facedatatrainlabels'

    # # Number of samples in the dataset
    # n_samples_train = 10
    # n_samples_val = 301
    # n_samples_test = 150
    # min_value = 0.1

    # # Face image dataset shape
    # img_shape = (60, 70)

    # X_train, Y_train = getDataLabels(train_data_path, train_labels_path, n_samples_train, img_shape=img_shape)

    # X_train_np = np.array(X_train)
    # Y_train_np = np.array(Y_train)

    # Y_train_np_new = []
    # for y in Y_train_np:
    #     if y == 0:
    #         Y_train_np_new.append(np.array([0, 1]))
    #     else:
    #         Y_train_np_new.append(np.array([1, 0]))

    model = train(train_X, train_label)

    print(model[0].shape)

    # for i in range(4):
    #     predict(train_X[i], model)
    #     print(train_label[i])
    predict([0.99, 0.1], model) 