import __future__

import numpy as np
import tensorflow as tf
import math
from datetime import datetime

"""
Configs:
word_embedding_dim
embedding_max_norm

batch_size

cell_hidden_dim: the hidden layer dimension in recurrent cells

learning_rate, 
decay_steps, 
decay_rate

epoch
batch_train_stats_every
"""

class RNN_image_cap():
	'''
	features: image features (extracted by pretrained CNN)
	captions: ground truth captions
	config: hyperparameters and other configurations
	path: the path to store the trained model
	'''
	def __init__(self, features, captions, word_to_idx, config, path=None):
		self.features = features
		self.captions = captions
		self.config = config
		self.path = path 

		self.word_to_idx = word_to_idx
		self.idx_to_word = {i: w for w, i in word_to_idx.items()}
		self.vocabulary_size = len(word_to_idx)
		self.caption_length = captions.shape[1]
		self.feature_size = features.shape[1]

		self._null = word_to_idx['<NULL>']
		self._start = word_to_idx.get('<START>', None)
		self._end = word_to_idx.get('<END>', None)

	'''
	transfer word indices to vectors by looking up the embedding matrix
	inputs:
		inputs: a batch of captions, shape = [batch_size, caption_length]

	outputs:
		vectors: the associated word vectors, 
				 shape = [batch_size, caption_length, word_embedding_dim]
	'''
	def word_idx_to_emb(self, inputs):
		V = self.vocabulary_size
		D = self.config.word_embedding_dim

		embeddings = tf.get_variable("word_embeddings", shape=[V,D])
		vectors = tf.nn.embedding_lookup(
			embeddings, inputs, max_norm=self.config.embedding_max_norm)

		return vectors

	'''
	Define the recurrent cell
	'''
	def _recurrent_cell(self):
		hidden_dim = self.config.cell_hidden_dim

		cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim)

		return cell

	'''
	Define the basic RNN
	inputs: batch_size X caption_length/time_steps X embedding_dim/input_dim
	initial_state: initial state of the RNN; here should be features of images
				   for hidden state and 0 for cell state
	rnn_outputs: batch_size X time_steps X hidden_dims
	'''
	def basic_RNN(self, inputs, initial_state):
		cell = self._recurrent_cell()

		rnn_outputs, states = tf.nn.dynamic_rnn(
			cell, inputs, initial_state=initial_state)

		return rnn_outputs, states


	'''
	Used to predict the next word
	inputs: rnn_outputs, batch_size X time_steps X hidden_dims
	outputs: batch_size X time_steps X vocabulary_size		
	'''
	def fully_connected(self, inputs):
		V = self.vocabulary_size

		outputs = tf.layers.dense(
				inputs, # input
				units=V,
				name='Output',
				)

		return outputs


	'''
	The whole graph of the RNN for image captioning
	'''
	def graph(self):
		cap_length = self.caption_length
		feature_size = self.feature_size
		batch_size = self.config.batch_size

		features = tf.placeholder(tf.float32, shape=[None, feature_size], name='features')
		captions = tf.placeholder(tf.int32, shape=[None, cap_length], name='captions')
		is_training = tf.placeholder(tf.bool, name='is_training')

		# Cut captions into two pieces: captions_in has everything but the last word
		# and will be input to the RNN; captions_out has everything but the first
		# word and this is what we will expect the RNN to generate. These are offset
		# by one relative to each other because the RNN should produce word (t+1)
		# after receiving word t. The first element of captions_in will be the START
		# token, and the first element of captions_out will be the first word.
		captions_in = captions[:, :-1]
		captions_out = captions[:, 1:]
		mask = (captions_out != self._null)

		initial_cell_states = tf.zeros_like(features)
		initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_states, features)

		embeddings = self.word_idx_to_emb(captions_in)
		rnn_outputs, _ = self.basic_RNN(embeddings, initial_state)
		logits = self.fully_connected(rnn_outputs)

			# batch_size X time_step(cap_length - 1)
		loss = tf.losses.sparse_softmax_cross_entropy(
			labels=captions_out, logits=logits, weights=mask)

		global_step = tf.Variable(0, trainable=False)

		## learning rate
		learning_rate = tf.train.exponential_decay(
			self.config.learning_rate, 
			global_step,
			self.config.decay_steps, 
			self.config.decay_rate
			)

		## the optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate)
		updates = optimizer.minimize(loss, global_step=global_step)

		placeholder_dict = {'features':features, 'captions':captions, 'is_training':is_training}
		fetch_dict = {
			'loss':loss,
			'updates': updates,
			'global_step': global_step,
			'learning_rate': learning_rate
			}

		return placeholder_dict, fetch_dict



	'''
	train the model
	'''
	def train(self):
		features = self.features
		captions = self.captions
		batch_size = self.config.batch_size

		# shuffle indicies
		train_indicies = np.arange(features.shape[0])
		# print(train_indicies)
		np.random.shuffle(train_indicies)

		placeholder_dict, fetch_dict = self.graph()
		feat = placeholder_dict['features']
		cap = placeholder_dict['captions']
		is_training = placeholder_dict['is_training']

		loss = fetch_dict['loss']
		updates = fetch_dict['updates']
		global_step = fetch_dict['global_step']
		train_fetch = [updates, loss, global_step]

		is_batch_train_stats = self.config.batch_train_stats_every > 0
		is_print = self.config.print_every > 0

		if(is_batch_train_stats):	
			now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
			root_logdir = "tf_logs"
			logdir = "{}/batch_train_stats-{}/".format(root_logdir, now)
			batch_train_summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
			loss_summary = tf.summary.scalar('LOSS', loss)

		# the saver object for saving check points
		saver = tf.train.Saver()

		with tf.Session() as sess:
			# initialize
			sess.run(tf.global_variables_initializer())

			for e in range(self.config.epoch):
				# make sure we iterate over the dataset once
				for i in range(int(math.ceil(features.shape[0]/batch_size))):
					# generate indicies for the batch
					start_idx = (i*batch_size)%features.shape[0]
					idx = train_indicies[start_idx:start_idx+batch_size]

					train_feed_dict = {
						feat: features[idx,:],
						cap: captions[idx],
						is_training: True
					}	

					_, batch_train_loss, step = sess.run(
						train_fetch, feed_dict=train_feed_dict)  

					if(is_batch_train_stats and step % self.config.batch_train_stats_every == 0):						
						summary = sess.run(loss_summary, feed_dict=train_feed_dict)
						batch_train_summary_writer.add_summary(summary, step) 

					iter_cnt = step - 1									
					# print stats for every n iteration
					if(is_print and iter_cnt % self.config.print_every == 0):						
						print("Iteration {0}: with minibatch training loss = {1:.3g}"
					  .format(iter_cnt, batch_train_loss))

			if is_batch_train_stats:
				batch_train_summary_writer.close()    



			# save the final results
			if(self.path != None):
				saver.save(sess, self.path)    						





	