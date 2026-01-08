import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, Timer
import asyncio
import threading

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

import requests
requests.packages.urllib3.disable_warnings()
import ssl

class tt_um_sigmoid_8bit(tf.keras.layers.Layer):
    def __init__(self, dut, **kwargs):
        super(tt_um_sigmoid_8bit, self).__init__(**kwargs)
        self.dut = dut

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    def call(self, inputs):
        def batch_fetch_py(x_batch):
            x_np = x_batch.numpy()
            x_flat = x_np.flatten()
            y_flat = np.zeros_like(x_flat)
            
            # Run TensorFlow in a background thread 
            # if the model is called via cocotb.external()
            event = threading.Event()

            # Async task that Cocotb will run
            async def run_batch_task():
                for i, val in enumerate(x_flat):
                    val_clip = int(np.clip(float(val) * 16.0, -128, 127))
                    self.dut._log.info(f"Fetching value for input: {val_clip}")
                    y_flat[i] = await fetch_value(self.dut, val_clip)
                    self.dut._log.info(f"Received output: {y_flat[i]}")
                # Task finished
                event.set()

            # Schedule task in Cocotb scheduler
            cocotb.start_soon(run_batch_task())

            # Block the TensorFlow thread until the simulation task finishes
            event.wait()
            
            return y_flat.astype(np.float32).reshape(x_np.shape)

        y = tf.py_function(func=batch_fetch_py, inp=[inputs], Tout=tf.float32)
        y.set_shape(inputs.shape)
        return tf.clip_by_value(y, 0.0, 1.0)
    

@cocotb.test()
async def test_project(dut):
    dut._log.info("Start")

    # Clock 100 KHz
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    dut._log.info("Training model")

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
        model.add(tt_um_sigmoid_8bit(dut))
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
        

async def reset(dut):
    dut._log.info("Reset")
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

async def fetch_value(dut, ui_in):
    dut.ui_in.value = ui_in
    #dut.uio_in.value = 0  # Dummy

    # Wait for one clock cycle to see the output values
    await ClockCycles(dut.clk, 1)

    # Wait to update registers
    await Timer(10, unit="ns") 

    return dut.uo_out.value

async def test_value(dut, ui_in):
    expected_output = ui_in << 2
    expected_output += 128

    if ui_in <= -32:
        expected_output = 0
    elif ui_in >= 32:
        expected_output = 255

    dut._log.info(f"Test ui_in={ui_in}, expected uo_out={expected_output}")
    
    await reset(dut)

    dut.ui_in.value = ui_in
    #dut.uio_in.value = 0  # Dummy

    # Wait for one clock cycle to see the output values
    await ClockCycles(dut.clk, 1)

    # Wait to update registers
    await Timer(10, unit="ns") 

    assert dut.uo_out.value == expected_output, f"Expected uo_out to be {expected_output} for ui_in={ui_in}, but got {dut.uo_out.value}"

    dut._log.info(f"Passed for ui_in={ui_in}, received uo_out={dut.uo_out.value}")
