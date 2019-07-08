import tensorflow as tf
import numpy as np
from sys import argv

# specify path to training data and testing data

if len(argv) < 4:
    print("Usage: myproj.py train_x train_y seed")
    exit(1)

train_x_location = argv[1] #"x_train_perm3.csv"
train_y_location = argv[2] #"y_train_perm3.csv"
log = argv[3]
seed=7 #argv[3]
test_x_location = "x_test.csv"
test_y_location = "y_test.csv"

# set the random seeds to make sure your results are reproducible
#from numpy.random import seed
np.random.seed(seed)
#from tensorflow import set_random_seed
tf.set_random_seed(seed)

print("Reading training data")
x_train = np.loadtxt(train_x_location, dtype=np.float, delimiter=",")
y_train = np.loadtxt(train_y_location, dtype=np.int, delimiter=",")

m, n = x_train.shape # m training examples, each with n features
m_labels, = y_train.shape # m2 examples, each with k labels
l_min = y_train.min()

assert m_labels == m, "x_train and y_train should have same length."
assert l_min == 0, "each label should be in the range 0 - k-1."
k = y_train.max()+1

print(m, "examples,", n, "features,", k, "categories.")


# print("Pre processing x of training data")
# x_train = x_train / 1.0

# define the training model
model = tf.keras.models.Sequential([
	tf.keras.layers.Dense(30, activation=tf.keras.activations.relu,
                          input_shape=(n,), kernel_regularizer=tf.keras.regularizers.l2(0.001)), #regularize 900 weights
	tf.keras.layers.BatchNormalization(), # normalize 30 outputs per batch
	tf.keras.layers.Dense(10, activation=tf.keras.activations.relu), # reduce to 10 features, pca shows this captures 95% of the variance
	tf.keras.layers.GaussianNoise(1), # don't overfit since it's such a small training set
	tf.keras.layers.Dense(5, activation=tf.keras.activations.relu),
	tf.keras.layers.Dropout(.4), #drop 8/10 weights going into softmax
	tf.keras.layers.Dense(k, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500, batch_size=18)

print("Reading testing data")
x_test = np.loadtxt(test_x_location, dtype=np.float, delimiter=",")
y_test = np.loadtxt(test_y_location, dtype=np.float, delimiter=",")

m_test, n_test = x_test.shape
m_test_labels,  = y_test.shape
l_min = y_train.min()

assert m_test_labels == m_test, "x_test and y_test should have same length."
assert n_test == n, "train and x_test should have same number of features."

print(m_test, "test examples.")

# print("Pre processing testing data")
# x_test = x_test / 1.0

print(model.evaluate(x_test, y_test, batch_size=18, verbose=1))
with open(log, 'a') as fout:
	fout.write("{}\n".format(model.evaluate(x_test, y_test, batch_size=18, verbose=1)))


