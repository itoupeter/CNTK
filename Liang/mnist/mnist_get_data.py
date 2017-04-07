
from __future__ import print_function
import gzip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import struct
import sys
from urllib.request import urlretrieve

def load_data(src, cimg):
    print('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    print('Done')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))[0]
            if n != 0x3080000:
                raise Exception('Invalid file: unexpected magic number.')
            n = struct.unpack('>I', gz.read(4))[0]
            if n != cimg:
                raise Exception('Invalid file: expected {0} entries.'.format(cimg))
            (crow, ccol) = struct.unpack('>II', gz.read(8))
            if crow != 28 or ccol != 28:
                raise Exception('Invalid file: expected 28 rows/cols per image.')
            res = np.fromstring(gz.read(cimg * crow * ccol), dtype = np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, crow * ccol))

def load_label(src, cimg):
    print('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    print('Done')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))[0]
            if n != 0x1080000:
                raise Exception('Invalid file: unexpected magic number.')
            n = struct.unpack('>I', gz.read(4))[0]
            if n != cimg:
                raise Exception('Invalid file: expected {0} entries'.format(cimg))
            res = np.fromstring(gz.read(cimg), dtype = np.uint8)
    finally:
        os.remove(gzfname)
    return [e == range(10) for e in res]

def try_download(data_src, label_src, cimg):
    data = load_data(data_src, cimg)
    label = load_label(label_src, cimg)
    return data, label

train_features_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
train_labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
num_train_samples = 60000
print('Downloading train data')
train_features, train_labels = try_download(train_features_url, train_labels_url, num_train_samples)

test_features_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
test_labels_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
num_test_samples = 10000
print('Downloading test data')
test_features, test_labels = try_download(test_features_url, test_labels_url, num_test_samples)

train_features_file = 'train_features'
train_labels_file = 'train_labels'
np.save(train_features_file, train_features)
np.save(train_labels_file, train_labels)

test_features_file = 'test_features'
test_labels_file = 'test_labels'
np.save(test_features_file, test_features)
np.save(test_labels_file, test_labels)

# sample_id = 5001
# plt.imshow(train_features[sample_id, :].reshape(28, 28), cmap = 'gray_r')
# plt.axis('off')
# plt.show()
# print("Image Label: ", train_labels[sample_id, 0])
