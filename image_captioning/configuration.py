import __future__
# import argparse as ap
import json
import sys

"""
This file defines a class that can includes all hyper-parameters from
a json file
"""
class Config:

	'''
	Inputs: 
		file: a json file contains all hyperparameters
	'''
	def __init__(self, filename=None):
		# should first go to the dir where includes main.py
		try:
			with open(filename) as f:
				dat = f.read()
			hp = json.loads(dat)
		except (FileNotFoundError, TypeError) as e:
			hp = {}



		'''
		Optimizer related
		Adam optimizer and exponential learning rate decay
		'''
		# learning rate 
		self.learning_rate = hp.get('learning_rate', 5e-3)

		# decay steps
		self.decay_steps = hp.get('decay_steps', 1)

		# decay rate
		self.decay_rate = hp.get('decay_rate', 0.995)

		# track the learning rate for every n iters, 0 for not track
		self.track_lr_every = hp.get('track_lr_every', 0)

		# Adam hyperparam: beta1, beta2, epsilon
		self.adam_beta1 = hp.get('adam_beta1', 0.9)
		self.adam_beta2 = hp.get('adam_beta2', 0.999)
		self.adam_epsilon = hp.get('adam_epsilon', 1e-8)





		'''
		Training related
		'''		

		# epoch
		self.epoch = hp.get('epoch', 50)

		# batch size
		self.batch_size = hp.get('batch_size', 25)

		'''
		Initialization
		Three types provided: Xavier uniform and HE, default is Xavier uniform
		'''
		# weight initializer in conv layers
		self.conv_init = hp.get('conv_init', 'Xavier')

		# weight initializer in full connected layers
		self.fc_init = hp.get('fc_init', 'Xavier')


		'''
		Logging related
		'''
		# print stats for every n iters, 0 for not print
		self.print_every = hp.get('print_every', 10)

		# track batch train losses and accuracy to tensor board for every n iters, 0 for not track
		self.batch_train_stats_every = hp.get('batch_train_stats_every', 1)


		# do the validation (and save the stats to tensorboard) for every n iters, 0 for not doing
		self.val_every = hp.get('val_every', 0)

		# save check points for every n iters, 0 for not saving
		self.ckpt_every = hp.get('ckpt_every', 0)



		'''
		Structure related
		'''
		self.word_embedding_dim = hp.get('word_embedding_dim', 256)

		self.embedding_max_norm = hp.get('embedding_max_norm', None)

		self.cell_hidden_dim = hp.get('cell_hidden_dim', 512)


		
		# whether to use dropout
		self.dropout = hp.get('dropout', False)

		# dropout rate
		self.dropout_rate = hp.get('dropout_rate', 0.5)

		# whether to use l1 reg and/or l2 reg
		self.l1_reg = hp.get('l1_reg', False)
		self.l2_reg = hp.get('l2_reg', False)

		# l1 and l2 regularization
		self.reg_lambda1 = hp.get('reg_lambda1', 1.0)
		self.reg_lambda2 = hp.get('reg_lambda2', 1.0)


		# loss type (hinge or softmax)
		self.loss = hp.get('loss', 'hinge')
		if self.loss != 'hinge' and self.loss != 'softmax':
			sys.exit('loss should be hinge or softmax')

