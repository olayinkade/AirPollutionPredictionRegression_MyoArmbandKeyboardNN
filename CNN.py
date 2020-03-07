# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from myo_keyboard_classification import process_single_axis_gyro


def separate_training_test_data():
    data, labels = process_single_axis_gyro('y')
    data = np.array(data)

    test_size = int(0.1 * len(data))
    training_data = list(data[:, 0][:-test_size])
    training_labels = list(data[:, 1][:-test_size])
    test_data = list(data[:, 0][-test_size:])
    test_labels = list(data[:, 1][-test_size:])

    return training_data, training_labels, test_data, test_labels


training_x, training_y, test_x, test_y = separate_training_test_data()

# network parameters
n_hidden_layer_1 = 256
n_hidden_layer_2 = 256
n_hidden_layer_3 = 256
num_input = len(training_x[0])
num_classes = 5

# input and output setup for the network
x = tf.placeholder('float')
y = tf.placeholder('float', [None, num_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_layer_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_layer_2, n_hidden_layer_3])),
    'h3': tf.Variable(tf.random_normal([n_hidden_layer_3, n_hidden_layer_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer_3, num_classes]))
}
biases = {
    'b1': tf.Variable([n_hidden_layer_1]),
    'b2': tf.Variable([n_hidden_layer_2]),
    'b3': tf.Variable([n_hidden_layer_3]),
    'out': tf.Variable([num_classes])
}


# create models
def multilayer_perceptron(data):
    layer_1 = tf.add(tf.matmul(data, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    output_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return output_layer
