import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

class data:
    def __init__(self):
        self.inputs = []
        self.outputs = []

def parse_sample(df, begin, end):
    origin = df['<HIGH>'].iloc[end]
    inputs = df['<HIGH>'].iloc[begin:end].to_numpy() / origin
    outputs = df['<HIGH>'].iloc[end + 1:end + 6].to_numpy() / origin

    return inputs, outputs

def load(df, sampleCount, previousRange):
    whole = len(df['<HIGH>']) - previousRange - 6
    if sampleCount > whole or sampleCount < 1:
        sampleCount = whole;
    
    print(sampleCount, "/", whole, "samples loaded")

    dataset = data()

    for i in range(0, sampleCount):
        [inputs, outputs] = parse_sample(df, i, i + previousRange)
        dataset.inputs.append(np.array(inputs))
        dataset.outputs.append(np.array(outputs))

    return dataset

trainfile = pd.read_csv('C:/Users/dean/Git/Seayon/SeayonTrading/res/EURUSD.csv', sep='\t')
testfile = pd.read_csv('C:/Users/dean/Git/Seayon/SeayonTrading/res/EURUSD_test.csv', sep='\t')

previousRange = 1439

train = load(trainfile, -1, previousRange)
test = load(testfile, -1, previousRange)
print()

train_in = np.array(train.inputs)
train_out = np.array(train.outputs)
test_in = np.array(test.inputs)
test_out = np.array(test.outputs)

model = keras.Sequential([
    layers.Dense(24, activation='relu', input_shape=(previousRange,)),
    layers.Dense(24, activation='relu'),
    layers.Dense(5, activation='relu')
])

model.compile(
    loss='mse',
    optimizer='adam'
)

before = model.evaluate(test_in, test_out)

print("--------------")
model.fit(train_in, train_out, epochs=50)
print("--------------")

after = model.evaluate(test_in, test_out)

pred = model.predict(test_in)

print()
origin = testfile['<HIGH>'].iloc[previousRange]

print("expected:  ", test_out[0])
print("output: ", pred[0])
print()
print("original:  ", test_out[0] * origin)
print("-->")
print("predicted: ", pred[0] * origin)