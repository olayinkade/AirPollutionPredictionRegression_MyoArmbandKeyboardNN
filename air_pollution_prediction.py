import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import library
tf.disable_v2_behavior()

# Parameters
learning_rate = 0.001
training_epochs = 1001
display_step = 20

train_data_path = 'training_data.csv'
test_data_path = 'test_data.csv'


def linear_regression_polynomial(degree, is_multi_variable):
    raw_train_dataset = library.data_processing(train_data_path)

    X_d = pd.DataFrame(raw_train_dataset[['TEMP', 'PRES', 'DEWP']]).to_numpy() # Change the variables here to train using different values
    Y_d = pd.DataFrame(raw_train_dataset[['PM2.5']]).to_numpy()

    X = tf.placeholder(tf.float32, name='x')
    Y = tf.placeholder(tf.float32, name='y')

    b = tf.Variable(np.random.normal(), name='bias')
    y_pred = tf.Variable(tf.random_normal([1]), name='pred')
    w = tf.Variable(tf.random_normal([1]), name='weight')

    for pow_i in range(1, degree):
        y_pred = tf.add(tf.multiply(tf.pow(X, pow_i), w), y_pred)

    loss = tf.reduce_sum(tf.square(y_pred - Y))/(2*X_d.shape[0])
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        count = 0
        # Fit all training data
        for epoch in range(training_epochs):
            for (x, y) in zip(X_d, Y_d):
                sess.run(optimizer, feed_dict={X: x, Y: y})

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=",
                        "{:.9f}".format(sess.run(loss, feed_dict={X: X_d, Y: Y_d})),
                        "W=", sess.run(w), "b=", sess.run(b))
                if not is_multi_variable:
                    fig = plt.figure(figsize=(10, 10), dpi=100)
                    ax = fig.add_subplot(111)
                    ax.set_ylim(0, 1)
                    # ax.set_aspect('equal')
                    ax.plot(X_d, Y_d, 'ro', label='Original data')

                    ax.plot(X_d, y_pred.eval(
                        feed_dict={X: X_d}, session=sess), label='Fitted line')
                    ax.legend()
                    plt.show()
                    fig.savefig('plot22_{:05d}.png'.format(count), bbox_inches='tight', dpi=100)
                    count = count + 1
                    plt.close(fig)
        print("Optimization Finished!")
        training_cost = sess.run(loss, feed_dict={X: X_d, Y: Y_d})
        print("Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n')
        raw_test_dataset = library.data_processing(test_data_path)
        X_test_d = pd.DataFrame(raw_test_dataset[['TEMP', 'PRES', 'DEWP']]).to_numpy() # Change the variables here to train using different values
        Y_test_d = pd.DataFrame(raw_test_dataset[['PM2.5']]).to_numpy()
        print("Testing... (L2 loss Comparison)")
        testing_cost = sess.run(tf.reduce_sum(tf.pow(y_pred-Y, 2))/(2*X_test_d.shape[0]),
                                feed_dict={X: X_test_d, Y: Y_test_d})
        print("Testing cost=", testing_cost)
        print("Absolute l2 loss difference:", abs(training_cost - testing_cost))


def linear_regression_categorical():
    raw_train_dataset = library.data_processing(train_data_path)
    dummies = pd.get_dummies(pd.DataFrame(raw_train_dataset[['wd']]))
    X_d = dummies.to_numpy()
    Y_d = pd.DataFrame(raw_train_dataset[['PM2.5']]).to_numpy()

    X = tf.placeholder(tf.float32, name='x')
    Y = tf.placeholder(tf.float32, name='y')

    w = tf.Variable(np.random.normal(), name='weight')
    b = tf.Variable(np.random.normal(), name='bias')

    y_pred = tf.add(tf.multiply(X, w), b)

    loss = tf.reduce_sum(tf.square(y_pred - Y)) / (2 * X_d.shape[0])
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Fit all training data
        for epoch in range(training_epochs):
            for (x, y) in zip(X_d, Y_d):
                x = x.reshape(1, X_d.shape[1])
                sess.run(optimizer, feed_dict={X: x, Y: y})

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=",
                      "{:.9f}".format(sess.run(loss, feed_dict={X: X_d, Y: Y_d})),
                      "W=", sess.run(w), "b=", sess.run(b))

                fig = plt.figure(figsize=(10, 10), dpi=100)
                ax = raw_train_dataset.plot.scatter(x='wd', y='PM2.5')
                ax.set_ylim(0, 1)

                ax.plot(X_d, sess.run(w) * X_d + sess.run(b), label='Fitted line')
                ax.legend()
                plt.show()
                plt.close(fig)
        print("Optimization Finished!")
        training_cost = sess.run(loss, feed_dict={X: X_d, Y: Y_d})
        t_w = sess.run(w)
        t_b = sess.run(b)
        print("Training cost=", training_cost, "W=", t_w, "b=", t_b, '\n')

        raw_test_dataset = library.data_processing(test_data_path)
        X_test_d = pd.DataFrame(raw_test_dataset[['wd']]).to_numpy()
        dummies = pd.get_dummies(pd.DataFrame(raw_test_dataset[['wd']]))
        X_d = dummies.to_numpy()
        Y_test_d = pd.DataFrame(raw_test_dataset[['PM2.5']]).to_numpy()
        print("Testing... (L2 loss Comparison)")
        testing_cost = sess.run(tf.reduce_sum(tf.pow(y_pred - Y, 2)) / (2 * X_test_d.shape[0]),
                                feed_dict={X: X_d, Y: Y_test_d})
        print("Testing cost=", testing_cost)
        print("Absolute l2 loss difference:", abs(training_cost - testing_cost))


def simple_linear_regression():
    raw_train_dataset = library.data_processing(train_data_path)
    X_d = pd.DataFrame(raw_train_dataset[['TEMP']]).to_numpy()
    Y_d = pd.DataFrame(raw_train_dataset[['PM2.5']]).to_numpy()

    X = tf.placeholder(tf.float32, [X_d.shape[0], X_d.shape[1]], name='x')
    Y = tf.placeholder(tf.float32, name='y')

    w = tf.Variable(np.random.normal(), [None, X_d.shape[1]], name='weight')
    b = tf.Variable(np.random.normal(), name='bias')

    y_pred = tf.add(tf.multiply(X, w), b)

    loss = tf.reduce_sum(tf.square(y_pred - Y)) / (2 * X_d.shape[0])
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        count = 0
        # Fit all training data
        for epoch in range(training_epochs):
            for (x, y) in zip(X_d, Y_d):
                sess.run(optimizer, feed_dict={X: x, Y: y})

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=",
                      "{:.9f}".format(sess.run(loss, feed_dict={X: X_d, Y: Y_d})),
                      "W=", sess.run(w), "b=", sess.run(b))

                fig = plt.figure(figsize=(10, 10), dpi=100)
                ax = fig.add_subplot(111)
                ax.set_ylim(0, 1)
                ax.plot(X_d, Y_d, 'ro', label='Original data')

                ax.plot(X_d, sess.run(w) * X_d + sess.run(b), label='Fitted line')
                ax.legend()
                plt.show()
                fig.savefig( 'plot_{:05d}.png'.format(count), bbox_inches='tight', dpi=100)
                count = count + 1
                plt.close(fig)
        print("Optimization Finished!")
        training_cost = sess.run(loss, feed_dict={X: X_d, Y: Y_d})
        t_w = sess.run(w)
        t_b = sess.run(b)
        print("Training cost=", training_cost, "W=", t_w, "b=", t_b, '\n')

        raw_test_dataset = library.data_processing(test_data_path)
        X_test_d = pd.DataFrame(raw_test_dataset[['TEMP']]).to_numpy()
        Y_test_d = pd.DataFrame(raw_test_dataset[['PM2.5']]).to_numpy()
        print("Testing... (L2 loss Comparison)")
        testing_cost = sess.run(tf.reduce_sum(tf.pow(y_pred - Y, 2)) / (2 * X_test_d.shape[0]),
                                feed_dict={X: X_test_d, Y: Y_test_d})
        print("Testing cost=", testing_cost)
        print("Absolute l2 loss difference:", abs(training_cost - testing_cost))


def main():
    linear_regression_polynomial(10, True)


if __name__ == '__main__':
    main()