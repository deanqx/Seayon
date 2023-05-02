import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def sgd(train_in, train_out, epochs, model, loss_fn, optimizer):
    # Train the model using SGD
    for epoch in range(epochs):
        for step, (x_batch_train, y_batch_train) in zip(train_in, train_out):
            with tf.GradientTape() as tape:
                # Forward pass
                y_pred = model(x_batch_train, training=True)
                # Compute the loss
                loss = loss_fn(y_batch_train, y_pred)
            # Compute the gradients
            grads = tape.gradient(loss, model.trainable_weights)
            # Update the model parameters
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

model = keras.Sequential([
    layers.Dense(3, activation='sigmoid', input_shape=(2,)),
    layers.Dense(2, activation='sigmoid')
])

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.5)
)

inputs = np.array([[1.0, 0.0], [0.0, 1.0]])
outputs = np.array([[0.0, 1.0], [1.0, 0.0]])

w = np.array(model.get_weights(), dtype=object)

w[0] = np.array([[-0.704276, -0.719596, 0.240516], [-0.851741, -0.912961, 0.512192]])
w[2] = np.array([[-0.855159, 0.502976], [-0.026765, 0.068819], [-0.898984, 0.672475]])

model.set_weights(w)

print(w, "\n")

before = model.evaluate(inputs, outputs)
pred = model.predict(inputs)
print(pred)

ttest = keras.Sequential([
    layers.Dense(3, activation='sigmoid', input_shape=(2,)),
    layers.Dense(2, activation='sigmoid')
])
loss_fn = tf.keras.losses.MeanSquaredError
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
sgd(inputs, outputs, epochs=1, model=ttest, loss_fn=loss_fn, optimizer=optimizer)
# model.fit(inputs, outputs, epochs=1, batch_size=2)

after = model.evaluate(inputs, outputs)
pred = model.predict(inputs)
print(pred)