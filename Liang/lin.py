
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from cntk import Trainer, learning_rate_schedule, UnitType
from cntk.learner import sgd
from cntk.ops import *
from cntk.utils import get_train_eval_criterion, get_train_loss

# generate random separable samples
def generate_data(dim, num_classes):
    tmp = np.random.randint(size = (dim[0], dim[1]), low = 0, high = 2)
    Y = [e[0] * 2 + e[1] for e in tmp]
    tmp = (tmp.astype(np.float32) - .5) * 4 + 5
    X = np.random.randn(dim[0], dim[1]).astype(np.float32) + tmp

    labels = [l == range(num_classes) for l in Y]
    labels = np.asarray(np.vstack(labels), np.float32)
    return X, labels

num_features = 2
num_classes = 4

# plt.figure(1)
# colors = ['r' if e[0] == 1 else 'g' if e[1] == 1 else 'b' for e in labels]
# plt.scatter(features[:, 0], features[:, 1], color = colors)
# plt.show();

# build network
my_params = {'w': None, 'b': None}

def linear_layer(input_var, num_classes):
    num_features = input_var.shape[0]
    weight_param = parameter(shape = (num_features, num_classes))
    bias_param = parameter(shape = (num_classes))
    my_params['w'], my_params['b'] = weight_param, bias_param
    return times(input_var, weight_param) + bias_param

input = input_variable(2, np.float32)
label = input_variable(num_classes, np.float32)
z = linear_layer(input, num_classes)
loss = cross_entropy_with_softmax(z, label)
eval_error = classification_error(z, label)

learning_rate = 0.5
lr_schedule = learning_rate_schedule(learning_rate, UnitType.sample)
learner = sgd(z.parameters, lr_schedule)
trainer = Trainer(z, (loss, eval_error), [learner])

batch_size = 50
features, labels = generate_data((batch_size, num_features), num_classes)

plt.figure(0)
classes = [np.argmax(e) for e in labels]
colors = ['r' if e == 0 else 'g' if e == 1 else 'b' if e == 2 else 'y' for e in classes]
plt.scatter(features[:, 0], features[:, 1], color = colors)
plt.show();

for i in range(10000):
    features, labels = generate_data((batch_size, num_features), num_classes)
    trainer.train_minibatch({input: features, label: labels})

    # loss, error = get_train_loss(trainer), get_train_eval_criterion(trainer)
    # print('loss: {0:.4f}, error: {1:.4f}'.format(loss, error))

test_features, test_labels = generate_data((1000, num_features), num_classes)
test_error = trainer.test_minibatch({input: test_features, label: test_labels})
print(test_error)

out = softmax(z)
result = out.eval({input: test_features})

plt.figure(0)
predictions = [np.argmax(e) for e in result[0, :, :]]
colors = ['r' if e == 0 else 'g' if e == 1 else 'b' if e == 2 else 'y' for e in predictions]
plt.scatter(test_features[:, 0], test_features[:, 1], color = colors)
plt.show();
