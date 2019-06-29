# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag','Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images_4d = np.reshape(train_images, (60000,28,28,1))
test_images_4d = np.reshape(test_images, (10000,28,28,1))

# MLP
given_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

given_model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("GIVEN MODEL MLP\n")
given_model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = given_model.evaluate(test_images, test_labels)
print('Test accuracy %f, loss %f\n' %(test_acc, test_loss))

# CNN
conv_model1 = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu, input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

conv_model1.compile(optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

print("FIRST CNN single convolution layers\n")
conv_model1.fit(train_images_4d, train_labels, epochs=5)
conv_loss, conv_acc = conv_model1.evaluate(test_images_4d, test_labels)
print("Conv1 accuracy %f, loss %f\n" % (conv_acc, conv_loss))

# CNN2
conv_model2 = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu, input_shape=(28,28,1)),
    keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

conv_model2.compile(optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

print("SECOND CNN b2b convolution layers\n")
conv_model2.fit(train_images_4d, train_labels, epochs=5)

conv_loss, conv_acc = conv_model2.evaluate(test_images_4d, test_labels)
print("Conv2 accuracy %f, loss %f\n" % (conv_acc, conv_loss))

# CNN with batch norm
bn_model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu, input_shape=(28,28,1)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

bn_model.compile(optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

print("CNN 3 batch norm\n")
bn_model.fit(train_images_4d, train_labels, epochs=5, validation_data=(test_images_4d,test_labels))

#conv_loss, conv_acc = conv_model3.evaluate(test_images_4d, test_labels)
#print("bn_model accuracy %f, loss %f\n" % (conv_acc, conv_loss))

# CNN with dropout
do_rate = 0.1
do_model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu, input_shape=(28,28,1)),
    keras.layers.Dropout(rate=do_rate),
    keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.Dropout(rate=do_rate),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.Dropout(rate=do_rate),
    keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.Dropout(rate=do_rate),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.Dropout(rate=do_rate),
    keras.layers.Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    keras.layers.Dropout(rate=do_rate),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

do_model.compile(optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

print("CNN 4 drop out\n")
do_model.fit(train_images_4d, train_labels, epochs=5, validation_data=(test_images_4d,test_labels))


