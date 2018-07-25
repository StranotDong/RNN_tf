import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.rnn_layers import *
from cs231n.captioning_solver import CaptioningSolver
from cs231n.classifiers.rnn import CaptioningRNN
from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from cs231n.image_utils import image_from_url

from configuration import Config
from RNN_image_cap import RNN_image_cap


# Load COCO data from disk; this returns a dictionary
# We'll work with dimensionality-reduced features for this notebook, but feel
# free to experiment with the original features by changing the flag below.
data = load_coco_data(max_train=50, pca_features=True)
captions = data['train_captions']
image_idxs = data['train_image_idxs']
image_features = data['train_features'][image_idxs]
word_to_idx = data['word_to_idx']

# Print out all the keys and values from the data dictionary
# for k, v in data.items():
#     if type(v) == np.ndarray:
#         print(k, type(v), v.shape, v.dtype)
#     else:
#         print(k, type(v), len(v))


config = Config()
# print(vars(config).items())
path = 'trained_model/test.ckpt'
model = RNN_image_cap(
	image_features, captions, word_to_idx, config, path=path)

model.train()

