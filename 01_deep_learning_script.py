import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

activation_layer = layers.Activation('sigmoid')
x = tf.linspace(-30.0, 30.0, 100)
y = activation_layer(x)

fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(x, y)
plt.grid()
plt.show()

# Check point - WORKs. Importing packages especially tensorflow gives errors that can be ignored if no GPU accel. is required. 


# Hypoeberlic target
activation_layer = layers.Activation('tanh')
x = tf.linspace(-30.0, 30.0, 100)
y = activation_layer(x)

fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(x, y)
plt.grid()
plt.show()

# Rectifier
activation_layer = layers.Activation('relu')
x = tf.linspace(-3.0, 3.0, 100)
y = activation_layer(x)

fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(x, y)
plt.grid()
plt.show()