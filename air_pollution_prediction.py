import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
tf.disable_v2_behavior()

# Parameters
learning_rate = 0.003
training_epochs = 500
display_step = 20

def data_processing(location):
    column_names = ['PM2.5', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']
    raw_dataset = pd.read_csv\
        ( location,
         names=column_names, na_values = "?", comment='\t',
         sep=",", skipinitialspace=True)
    wd = pd.DataFrame(raw_dataset[['wd']])
    x = raw_dataset[['PM2.5', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']].values.astype(float)

    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x)

    # Run the normalizer on the dataframe
    df_normalized = pd.DataFrame(x_scaled)
    raw_dataset = pd.concat([df_normalized, wd], axis=1)
    raw_dataset.columns = ['PM2.5', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'wd']
    return raw_dataset

raw_train_dataset = data_processing(r'C:\Users\OBTAdelakun\Desktop\AirPollutionPredictionRegression_MyoArmbandKeyboardNN\training_data.csv')

X_d = pd.DataFrame(raw_train_dataset[['TEMP']]).to_numpy()
Y_d = pd.DataFrame(raw_train_dataset[['PM2.5']]).to_numpy()

X = tf.placeholder(tf.float32, name='x')
Y = tf.placeholder(tf.float32, name='y')

w = tf.Variable(np.random.normal(), name='weight')
b = tf.Variable(np.random.normal(), name='bias')

y_pred = tf.add(tf.multiply(w, X), b)

loss = tf.reduce_sum(tf.square(y_pred - Y))/(2*X_d.shape[0])
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    count=0
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
            # ax.set_aspect('equal')
            ax.plot(X_d, Y_d, 'ro', label='Original data')

            ax.plot(X_d, sess.run(w) * X_d + sess.run(b), label='Fitted line')
            ax.legend()
            plt.show()
            fig.savefig('plot_{:05d}.png'.format(count), bbox_inches='tight', dpi=100)
            count = count + 1
            plt.close(fig)
    print("Optimization Finished!")
    training_cost = sess.run(loss, feed_dict={X: X_d, Y: Y_d})
    print("Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n')

    raw_test_dataset = data_processing(r'C:\Users\OBTAdelakun\Desktop\AirPollutionPredictionRegression_MyoArmbandKeyboardNN\test_data.csv')
    X_test_d = pd.DataFrame(raw_test_dataset[['TEMP']]).to_numpy()
    Y_test_d = pd.DataFrame(raw_test_dataset[['PM2.5']]).to_numpy()
    print("Testing... (L2 loss Comparison)")
    testing_cost = sess.run(tf.reduce_sum(tf.pow(y_pred-Y, 2))/(2*X_test_d.shape[0]),
                            feed_dict={X: X_test_d, Y: Y_test_d})
    print("Testing cost=", testing_cost)
    print("Absolute l2 loss difference:", abs(training_cost - testing_cost))
