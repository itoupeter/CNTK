
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import cntk as C
from cntk import Trainer, learning_rate_schedule, momentum_schedule, UnitType
from cntk.learner import sgd, fsadagrad
from cntk.initializer import glorot_uniform
from cntk.layers import default_options, Input, Dense

np.random.seed(0)

def generate_random_data_sample(sample_size, num_features, num_classes):
    dim = (sample_size, num_features)
    tmp = np.random.randint(size = (dim[0], dim[1]), low = 0, high = 2)
    Y = [e[0] * 2 + e[1] for e in tmp]
    tmp = (tmp.astype(np.float32) - .5) * 4
    X = np.random.randn(dim[0], dim[1]).astype(np.float32) + tmp

    labels = [l == range(num_classes) for l in Y]
    labels = np.asarray(np.vstack(labels), np.float32)
    return X, labels

def linear_layer(input_var, num_classes):
    num_features = input_var.shape[0]
    weight = C.parameter(shape = (num_features, num_classes))
    bias = C.parameter(shape = (num_classes))
    return C.times(input_var, weight) + bias

def dense_layer(input_var, num_classes, nonlinearity):
    return nonlinearity(linear_layer(input_var, num_classes))

num_features = 2
num_classes = 4
hidden_layer_dim = 20

sample_size = 1000
features, labels = generate_random_data_sample(sample_size, num_features, num_classes)

plt.figure(0)
color_list = ['r', 'b', 'g', 'y']
colors = [color_list[np.argmax(e)] for e in labels]
plt.scatter(features[:, 0], features[:, 1], c = colors, marker = 'x')
plt.title('Input')
# plt.show();

input = Input(num_features)
label = Input(num_classes)
h1 = dense_layer(input, hidden_layer_dim, C.sigmoid)
h2 = dense_layer(h1, hidden_layer_dim, C.sigmoid)
z = linear_layer(h2, num_classes)

loss = C.cross_entropy_with_softmax(z, label)
error = C.classification_error(z, label)
learning_rate = 0.2
momentum = 0.9
lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
m_schedule = momentum_schedule(momentum)
learner = fsadagrad(z.parameters, lr_schedule, m_schedule)
trainer = Trainer(z, (loss, error), [learner])

minibatch_size = 50
sample_size = 80000
num_minibatches = sample_size / minibatch_size

for i in range(int(num_minibatches)):
    features, labels = generate_random_data_sample(minibatch_size, num_features, num_classes)
    trainer.train_minibatch({input: features, label: labels})

test_features, test_labels = generate_random_data_sample(1000, num_features, num_classes)
print(trainer.test_minibatch({input: test_features, label: test_labels}))

out = C.softmax(z)
predictions = out.eval({input: test_features})

plt.figure(1)
colors = [color_list[np.argmax(e)] for e in predictions[0, :, :]]
plt.scatter(test_features[:, 0], test_features[:, 1], c = colors, marker = 'x')
plt.title('Output')
plt.show()
