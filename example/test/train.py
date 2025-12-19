import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

import requests
requests.packages.urllib3.disable_warnings()
import ssl



class tt_um_sigmoid_8bit(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(tt_um_sigmoid_8bit, self).__init__(**kwargs)

    def call(self, inputs):
        # TODO: Implement cocotb testbench integration here.
        
        # Simulate the ASIC behavior: y = 0.25x + 0.5, clipped to [0, 1]
        x = inputs
        y = 0.25 * x + 0.5
        y = tf.clip_by_value(y, 0.0, 1.0)
        return y


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

if not os.path.exists("mnist_model.keras"):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128))
    model.add(tt_um_sigmoid_8bit())
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_images, train_labels, epochs=5)

    model.save("mnist_model.keras")
else:
    model = tf.keras.models.load_model("mnist_model.keras")

loss, accuracy = model.evaluate(test_images, test_labels)
print(f"{accuracy=}")

prediction = model.predict(test_images)
plt.imshow(test_images[0], cmap="gray")
num = str(np.argmax(prediction[0]))
plt.text(0, 0.5, "Model predicted: " + str(num), backgroundcolor="white", fontweight="bold")
plt.show()


