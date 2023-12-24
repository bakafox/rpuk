import numpy as np

def sigmoid(x): # сигмоидная фукнция, ограничивает результат в диапазоне 0..1
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): # производная от сигмоидной функции
    return x * (1 - x)

def loss(t, y): # простая функция потерь
    return t - y


class Neuron:
    def __init__(self, input_size, lr):
        self._weights = np.random.uniform(size=(input_size,))
        self._bias = np.random.uniform()
        self._lr = lr

    def forward(self, x):
        self._input = x
        self._output = sigmoid(np.dot(x, self._weights) + self._bias)
        return self._output

    def backward(self, loss):
        output_delta = sigmoid_derivative(self._output) * loss
        self._weights += (self._input * output_delta) * self._lr 
        self._bias += (output_delta) * self._lr

class Model:
    def __init__(self, input_size, hidden_size, lr):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._lr = lr

        self.hidden_layer = [] # по условию их ровно два, но так интереснее
        for i in range(hidden_size):
            self.hidden_layer.append(Neuron(input_size, lr))

        self.output_layer = [Neuron(hidden_size, lr)] # по условию

    def forward(self, x):
        self.hidden_layer_output = []
        for i in range(self._hidden_size):
            self.hidden_layer_output.append(self.hidden_layer[i].forward(x))

        return self.output_layer[0].forward(
            np.array(self.hidden_layer_output).flatten()
        )

    def backward(self, x, loss):
        output_delta = sigmoid_derivative(self.output_layer[0]._output) * loss
        self.output_layer[0]._weights += (output_delta * self.output_layer[0]._input) * self._lr
        self.output_layer[0]._bias += (output_delta) * self._lr

        for i in range(self._hidden_size):
            hidden_delta = (output_delta * self.output_layer[0]._weights[i]
                        * sigmoid_derivative(self.hidden_layer[i]._output))
            self.hidden_layer[i]._weights += (hidden_delta * self.hidden_layer[i]._input) * self._lr
            self.hidden_layer[i]._bias += (hidden_delta) * self._lr


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])
model = Model(input_size=2, hidden_size=3, lr=1.2)

for epoch in range(10000):
    for i in range(len(inputs)):
        y = model.forward(inputs[i])
        err = loss(expected_output[i], y)
        model.backward(inputs[i], err)

print("+---+---+-----+---------------")
print("| A | B | XOR | Предсказание")
print("+---+---+-----+---------------")
for i in range(len(inputs)):
    print("|", inputs[i][0], "|", inputs[i][1],
          "|", expected_output[i][0], " ",
          "|", np.round(model.forward(inputs[i])[0]) == expected_output[i][0],
          ":", model.forward(inputs[i])[0])