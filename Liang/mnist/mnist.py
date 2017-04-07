
from __future__ import print_function
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

import cntk as C
from cntk import Trainer, learning_rate_schedule, UnitType
from cntk.io import CTFDeserializer, MinibatchSource, StreamDef, StreamDefs
from cntk.io import INFINITELY_REPEAT, FULL_DATA_SWEEP
from cntk.initializer import glorot_uniform
from cntk.layers import default_options, Input, Dense
from cntk.learner import sgd
from cntk.utils import get_train_eval_criterion, get_train_loss

def linear_layer(input_var, num_classes):
    num_features = input_var.shape[0]
    weight_param = C.parameter(shape = (num_features, num_classes), init = C.glorot_normal())
    bias_param = C.parameter(shape = (num_classes), init = C.glorot_normal())
    return C.times(input_var, weight_param) + bias_param

def dense_layer(input_var, num_classes, nonlinearity):
    return nonlinearity(linear_layer(input_var, num_classes))

train_features_file, train_labels_file = 'train_features.npy', 'train_labels.npy'
train_features, train_labels = np.load(train_features_file), np.load(train_labels_file)
train_features, train_labels = train_features.astype(np.float32), train_labels.astype(np.float32)

test_features_file, test_labels_file = 'test_features.npy', 'test_labels.npy'
test_features, test_labels = np.load(test_features_file), np.load(test_labels_file)
test_features, test_labels = test_features.astype(np.float32), test_labels.astype(np.float32)

# sample_id = 5850
# plt.figure(0)
# plt.imshow(train_features[sample_id, :].reshape(28, 28), cmap = 'gray_r')
# plt.show()
# print(train_labels[sample_id, 0]); exit()

# model
num_features = 784
num_classes = 10
hidden_layer_dim = 400

input = Input(num_features)
label = Input(num_classes)
h1 = dense_layer(input / 256, hidden_layer_dim, C.relu)
h2 = dense_layer(h1, hidden_layer_dim, C.relu)
z = linear_layer(h2, num_classes)

loss = C.cross_entropy_with_softmax(z, label)
error = C.classification_error(z, label)
learning_rate = 0.2
lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
learner = sgd(z.parameters, lr_schedule)
trainer = Trainer(z, (loss, error), [learner])

# training
num_sweeps = 13
minibatch_size = 50
num_train_samples = 60000
num_test_samples = 10000

for i in range(num_sweeps):
    for j in range(0, num_train_samples, minibatch_size):
        trainer.train_minibatch({input: train_features[j:j + minibatch_size, :], label: train_labels[j:j + minibatch_size, :]})
    train_error = get_train_eval_criterion(trainer)

    test_error = 0.
    for j in range(0, num_test_samples, minibatch_size):
        test_data = {input: test_features[j:j + minibatch_size, :], label: test_labels[j:j + minibatch_size, :]}
        test_error = test_error + trainer.test_minibatch(test_data)
    test_error = test_error / (num_test_samples / minibatch_size)

    print('sweep {0} train error: {1:.4f} test error: {2:.4f}'.format(i, train_error, test_error), flush = True)
