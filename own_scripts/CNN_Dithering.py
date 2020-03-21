from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from PIL import Image

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.datasets import mnist
from datetime import datetime
from tensorflow import keras
from packaging import version
from tqdm import tqdm
import numpy as np
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#class_names = ['zero', 'one', 'two', 'three', 'four', 'five',
#              'six', 'seven', 'eight', 'nine']
class_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
""" show 10 first  pictures """
fig = plt.figure()
st = plt.suptitle("Input pictures", fontsize=14)
st.set_y(1)

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])#, cmap=  plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.tight_layout()
plt.show()


""" dither pictures """
print("Pictures in trainset are dithered")
for i, pic in tqdm(enumerate(train_images)):
    picture_grey = Image.fromarray(pic)
    picture_dither = picture_grey.convert("1")
    train_images[i] = picture_dither
train_images = np.array(train_images)

print("Pictures in test set are dithered")
for i, pic in tqdm(enumerate(test_images)):
    picture_grey = Image.fromarray(pic)
    picture_dither = picture_grey.convert("1")
    test_images[i] = picture_dither
test_images = np.array(test_images)
""" show 10 first dither pictures """
fig = plt.figure()
st = plt.suptitle("Dither pictures", fontsize=14)
st.set_y(1)

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])#, cmap=  plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.tight_layout()
plt.show()

# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0
train_images=train_images.reshape((-1, 28, 28, 1))
test_images=test_images.reshape((-1, 28, 28, 1))
print("Shape of input {}".format(test_images.shape))


model = models.Sequential()
model.add(layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Define the Keras TensorBoard callback.
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

#print("TensorFlow version: ", tf.__version__)
#assert version.parse(tf.__version__).release[0] >= 2, \
   # "This notebook requires TensorFlow 2.0 or above."


#model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_images, train_labels, epochs=1,
                    validation_data=(test_images, test_labels),
                    callbacks=[tensorboard_callback])

plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
