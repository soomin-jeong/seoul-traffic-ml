from __future__ import division

import tensorflow as tf
import os

from process_input import make_test_set
from process_input import make_training_set

dir_path = os.path.dirname(os.path.realpath(__file__))
TRAINING_SET_FILE_NAME = dir_path + '/traffic_volume_train.csv'
TEST_SET_FILE_NAME = dir_path + '/traffic_volume_test.csv'

training_set, training_set_y = make_training_set(TRAINING_SET_FILE_NAME)
testing_set, testing_set_y = make_test_set(TEST_SET_FILE_NAME)

# Placeholder you can assign values in future its kind of a variable
#  v = ("variable type",None) -- You can assign any number of variables for v
#  v = ("variable type",4)    -- You can assign 4 variables for v
#  v = ("variable type",[None,4])  -- you can have multidimensional values here
# Here the no.of rows you can have any number but the columns are fixed with size 4
training_values = tf.placeholder("float", [None, len(training_set[0])])
test_values = tf.placeholder("float", [len(training_set[0])])

# This is the distance formula to calculate the distance between the test values and the training values
distance = tf.reduce_sum(tf.abs(tf.add(training_values, tf.negative(test_values))), reduction_indices=1)

# Returns the index with the smallest value across dimensions of a tensor
prediction = tf.arg_min(distance, 0)

# Initializing  the session
init = tf.initialize_all_variables()

# Starting the calculation process
# For every test sample, the above "distance" formula will get called and the distance formula will return the
# distances from the traning set values to the test sample and then the "prediction" will return the smallest
# distance index.

with tf.Session() as sess:
    sess.run(init)
    # Looping through the test set to compare against the training set
    errors = []

    for i in range(len(testing_set)):
        # Tensor flow method to get the prediction nearer to the test parameters from the training set.
        index_in_trainingset = sess.run(prediction,
                                        feed_dict={training_values: training_set, test_values: testing_set[i]})
        errors.append(round(abs((int(training_set_y[index_in_trainingset])-int(testing_set_y[i]))/int(testing_set_y[i])*100),2))

        print("Test %d, prediction: %s / correct_value: %s [%s %%]" % (i, training_set_y[index_in_trainingset],
                                                                    testing_set_y[i],
                                                                    errors[i]))

    avg_error = sum(errors)/len(testing_set)
    print("Avg. error: %s%%" % (round(avg_error,2)))