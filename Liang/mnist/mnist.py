
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

train_features_file, train_labels_file = 'train_features.npy', 'train_labels.npy'
train_features, train_labels = np.load(train_features_file), np.load(train_labels_file)

test_features_file, test_labels_file = 'test_features.npy', 'test_labels.npy'
test_features, test_labels = np.load(test_features_file), np.load(test_labels_file)

sample_id = 5832
plt.figure(0)
plt.imshow(train_features[sample_id, :].reshape(28, 28), cmap = 'gray_r')
plt.show()
print(train_labels[sample_id, 0])
