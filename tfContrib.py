from Data.datacontrol import CSVData
import tensorflow as tf
import os
import numpy as np

PATH = os.path.dirname(os.path.realpath(__file__))


data = CSVData()
#data.updateData()
data.generateCSV()

training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=PATH+"/Data/train.csv",
    target_dtype=np.float32,
    features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=PATH+"/Data/test.csv",
    target_dtype=np.float32,
    features_dtype=np.float32)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=90)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
                                            hidden_units=[50, 100, 50],
                                            model_dir=PATH+"/tmp/iris_model")

# Define the training inputs
print("Training Data: ")
print(training_set.data)
print(training_set.target)
def get_train_inputs():
  x = tf.constant(training_set.data)
  y = tf.constant(training_set.target)

  return x, y

# Fit model.
print("start fit")
regressor.fit(input_fn=get_train_inputs, steps=2000)
print("finished")

# Define the test inputs
def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y

# Evaluate accuracy.
accuracy_score = regressor.evaluate(input_fn=get_test_inputs, steps=1)

test_results = regressor.predict_scores(input_fn=get_test_inputs, as_iterable=False)

print(test_results)
print(test_results.dtype, len(test_results))
print(test_set.target)
print(test_set.target.dtype, len(test_set.target), sum(test_set.target)/34)
#acc = tf.contrib.metrics.accuracy(test_results, test_set.target) # Not for floats
#print(acc)
print(accuracy_score)

# Classify two new flower samples.
# def new_samples():
# return np.array(
#   [[6.4, 3.2, 4.5, 1.5],
#    [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
#
# predictions = list(regressor.predict(input_fn=new_samples))
#
# print(
#   "New Samples, Class Predictions:    {}\n"
#   .format(predictions))
#
# if __name__ == "__main__":
#     main()
