
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import cntk as C
from cntk import Trainer, learning_rate_schedule, UnitType
from cntk.learner import sgd
from cntk.initializer import glorot_uniform
from cntk.layers import default_options, Input, Dense

np.random.seed(0)

def generate_random_data_sample(sample_size, num_features, num_classes):
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)
    X = (np.random.randn(sample_size, num_features) + 3) * (Y + 1)
    X = X.astype(np.float32)
    class_ind = [Y==class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y

def linear_layer(input_var, num_classes):
    num_features = input_var.shape[0]
    weight = C.parameter(shape = (num_features, num_classes))
    bias = C.parameter(shape = (num_classes))
    return C.times(input_var, weight) + bias

def dense_layer(input_var, num_classes, nonlinearity):
    return nonlinearity(linear_layer(input_var, num_classes))

sample_size = 64
num_features = 2
num_classes = 2
hidden_layer_dim = 50
features, labels = generate_random_data_sample(sample_size, num_features, num_classes)

plt.figure(0)
colors = ['r' if e[0] == 1 else 'b' for e in labels]
plt.scatter(features[:, 0], features[:, 1], c = colors, marker = 'x')
# plt.show();

input = Input(num_features)
label = Input(num_classes)
h1 = dense_layer(input, hidden_layer_dim, C.sigmoid)
h2 = dense_layer(h1, hidden_layer_dim, C.sigmoid)
z = linear_layer(h2, num_classes)

loss = C.cross_entropy_with_softmax(z, label)
error = C.classification_error(z, label)
learning_rate = 0.5
lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
learner = sgd(z.parameters, lr_schedule)
trainer = Trainer(z, (loss, error), [learner])

minibatch_size = 50
sample_size = 20000
num_minibatches = sample_size / minibatch_size

for i in range(int(num_minibatches)):
    features, labels = generate_random_data_sample(minibatch_size, num_features, num_classes)
    trainer.train_minibatch({input: features, label: labels})

test_features, test_labels = generate_random_data_sample(1000, num_features, num_classes)
print(trainer.test_minibatch({input: test_features, label: test_labels}))

out = C.softmax(z)
predictions = out.eval({input: test_features})

plt.figure(1)
colors = ['r' if np.argmax(e) == 0 else 'b' for e in predictions[0, :, :]]
plt.scatter(test_features[:, 0], test_features[:, 1], c = colors, marker = 'x')
plt.show()
