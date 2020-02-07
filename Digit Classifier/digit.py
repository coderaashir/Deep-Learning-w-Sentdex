import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist # 28x28 images of handwritten digits: 0-9, downloaded 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

x_test = tf.keras.utils.normalize(x_test, axis = 1) # scales data from 0...255 scale to 0..1 scale
x_train = tf.keras.utils.normalize(x_train, axis = 1)

model = tf.keras.models.Sequential() 
model.add(tf.keras.layers.Flatten()) # input layer

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # dense layer - 128 neurons, tf.nn.relu activation function
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print (val_loss, val_acc) 

predictions = model.predict([x_test]) 
print np.argmax(predictions[0])

plt.imshow(x_test[0])
plt.show() 