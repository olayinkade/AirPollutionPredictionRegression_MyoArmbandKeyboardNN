# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np

from myo_keyboard_classification import process_single_axis_gyro

LEARNING_RATE = 0.001
TRAINING_EPOCHS = 1000
BATCH_SIZE = 4
DISPLAY_STEP = 200


# seperate given data to training data and test data
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
n_hidden_layer_1 = 75
n_hidden_layer_2 = 75
n_hidden_layer_3 = 75
n_hidden_layer_4 = 75
num_input = len(training_x[0])
num_classes = 5

# input and output setup for the network
x = tf.placeholder('float', [None, num_input])
y = tf.placeholder('float', [None, num_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_layer_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_layer_1, n_hidden_layer_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_layer_2, n_hidden_layer_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_layer_3, n_hidden_layer_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer_4, num_classes]))
}
biases = {
    'b1': tf.Variable([n_hidden_layer_1]),
    'b2': tf.Variable([n_hidden_layer_2]),
    'b3': tf.Variable([n_hidden_layer_3]),
    'b4': tf.Variable([n_hidden_layer_4]),
    'out': tf.Variable([num_classes])
}


# create models
def multilayer_perceptron(data):
    biases['b1'] = tf.cast(biases['b1'], tf.float32)
    biases['b2'] = tf.cast(biases['b2'], tf.float32)
    biases['b3'] = tf.cast(biases['b3'], tf.float32)
    biases['b4'] = tf.cast(biases['b4'], tf.float32)
    biases['out'] = tf.cast(biases['out'], tf.float32)
    layer_1 = tf.add(tf.matmul(data, weights['h1']), biases['b1'])
    layer_1 = tf.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.sigmoid(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.sigmoid(layer_3)
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.sigmoid(layer_4)
    output_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return output_layer


def train_network(data):
    model = multilayer_perceptron(data)
    # define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss)
    # initialize variables
    init = tf.global_variables_initializer()

    # start TensorFlow session
    with tf.Session() as session:
        session.run(init)
        # training cycle
        for epoch in range(TRAINING_EPOCHS):
            avg_cost = 0.0
            total_batch = int(len(training_x) / BATCH_SIZE)
            # loop over all batches
            for i in range(total_batch):
                batch_x = np.array(training_x[i*BATCH_SIZE: (i + 1)*BATCH_SIZE])
                batch_y = np.array(training_y[i*BATCH_SIZE: (i + 1)*BATCH_SIZE])
                _, c = session.run([train_op, loss], feed_dict={x: batch_x, y: batch_y})
                # compute average cost
                avg_cost += c / total_batch

            # display cost per epoch step
            if epoch % DISPLAY_STEP == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))

        # test the model
        # predictions = tf.nn.softmax(model)  # apply softmax to model
        # correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
        # # calculate accuracy
        # accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
        correct_predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
        output_weight = session.run(weights['out'])
        output_bias = session.run(biases['out'])

        for i in range(len(test_x)):
            print('Expecting: {}'.format(test_y[i]))
            output = model.eval(feed_dict={x: [test_x[i]]})
            # print(model.eval(feed_dict={x: [test_x[i]]}))

            softmax = tf.nn.softmax(output).eval()
            print((tf.nn.softmax(output).eval()))

        print('Accuracy: ', accuracy.eval({x: test_x, y: test_y}) * 100, '%')


def main(argv=None):
    train_network(x)


if __name__ == '__main__':
    main()
