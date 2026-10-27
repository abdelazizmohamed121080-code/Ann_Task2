
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ds =pd.read_csv("penguins.csv")
class preprocess:
    def __init__(self, ds):
        ds = ds.dropna()
        ds['Species'] = ds['Species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
        ds['OriginLocation'] = ds['OriginLocation'].map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})

        x = ds.drop(columns=['Species']).to_numpy()
        y = ds['Species'].to_numpy()

        indices = np.random.permutation(len(x))   # array of shuffled row numbers
        split = int(0.8 * len(x))                 # ONE integer

        x_trainindc = indices[:split]             # array
        x_testindc = indices[split:]              # array

        self.x_train = x[x_trainindc]             # array index: no colon
        self.y_train = y[x_trainindc]
        self.x_test = x[x_testindc]
        self.y_test = y[x_testindc]

        print(len(x))

data = preprocess(ds)
print(data.x_train.shape, data.x_test.shape)


class Layer:
    def __init__(self, inputs, neurons):
        self.inputs = inputs
        self.neurons = neurons
        self.biases = np.random.randn(neurons) * 0.1
        self.weights = np.random.randn(inputs, neurons) * 0.1

    def forward(self, inputs):
        self.forwardoutputs = np.dot(inputs, self.weights) + self.biases
        return self.forwardoutputs


class Sigmoid_activation:
    def forward(self, inputs, slope=1.0):
        self.sigoutputs = 1 / (1 + np.exp(-slope * inputs))
        return self.sigoutputs


class Tanh_activation:
    def forward(self, inputs, slope=1.0):
        self.Tanhoutput = (1 - np.exp(-slope * inputs)) / (1 + np.exp(-slope * inputs))
        return self.Tanhoutput


layer1 = Layer(5, 2)
v=layer1.forward(5)
activation1 = Sigmoid_activation()
activation2 = Tanh_activation()
outputs = activation1.forward(v,slope=0.5)
output1=activation2.forward(v,slope=0.5)
print(output1)
plt.plot(data.x_train)
plt.plot(outputs)
plt.plot(output1)
plt.plot(data.x_test)
plt.show()