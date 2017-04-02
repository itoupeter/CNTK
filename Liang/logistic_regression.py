from __future__ import print_function
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from cntk import Trainer, learning_rate_schedule, UnitType
from cntk.learner import sgd
from cntk.ops import *
from cntk.utils import get_train_eval_criterion, get_train_loss

# assert 'TEST_DEVICE' in os.environ
# assert os.environ['TEST_DEVICE'] == 'gpu'

input_dim = 2
num_output_classes = 2

np.random.seed(0)

def generate_random_data_sample(sample_size, feature_dim, num_classes):
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    X = (np.random.randn(sample_size, feature_dim) + 3) * (Y + 1)

    X = X.astype(np.float32)

    class_ind = [Y==class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y

mysamplesize = 32
features, labels = generate_random_data_sample(mysamplesize, input_dim, num_output_classes)

# visualize data
# import matplotlib.pyplot as plt
#
# colors = ['r' if l == 0 else 'b' for l in labels[:, 0]]
# plt.scatter(features[:, 0], features[:, 1], c=colors)
# plt.xlabel("Scaled age (in yrs)")
# plt.ylabel("Tumor size (in cm)")
# plt.show()

input = input_variable(input_dim, np.float32)
mydict = {"w":None, "b":None}

def linear_layer(input_var, output_dim):
    input_dim = input_var.shape[0]
    weight_param = parameter(shape=(input_dim, output_dim))
    bias_param = parameter(shape=(output_dim))
    mydict['w'], mydict['b'] = weight_param, bias_param
    return times(input_var, weight_param) + bias_param

output_dim = num_output_classes
z = linear_layer(input, output_dim)
label = input_variable((num_output_classes), np.float32)
loss = cross_entropy_with_softmax(z, label)
eval_error = classification_error(z, label)

learning_rate = 0.5
lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
learner = sgd(z.parameters, lr_schedule)
trainer = Trainer(z, (loss, eval_error), [learner])

def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

def print_training_progress(trainer, mb, frequency, verbose=True):
    training_loss, eval_error = "NA", "NA"

    if mb % frequency == 0:
        training_loss = get_train_loss(trainer)
        eval_error = get_train_eval_criterion(trainer)
        if verbose:
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}".format(mb, training_loss, eval_error))

    return mb, training_loss, eval_error

minibatch_size = 25
num_samples_to_train = 20000
num_minibatches_to_train = int(num_samples_to_train / minibatch_size)

training_progress_output_freq = 50

plotdata = {"batchsize":[], "loss":[], "error":[]}

for i in range(0, num_minibatches_to_train):
    features, labels = generate_random_data_sample(minibatch_size, input_dim, num_output_classes)
    trainer.train_minibatch({input:features, label:labels})
    batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)

    if not (loss == "NA" or error == "NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)

plotdata["avgloss"] = moving_average(plotdata["loss"])
plotdata["avgerror"] = moving_average(plotdata["error"])

plt.figure(1)
plt.subplot(211)
plt.plot(plotdata['batchsize'], plotdata["avgloss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')

plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error')
plt.show()
