# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import options
flags = options.get()

class A3CModel(object):
	def get_vars(self):
		return self.variables # get model variables
		
	def get_action_vector(self, action): # transform action into a 1-hot-vector (if softmax is in use)
		if self.softmax_dimension == 0:
			return action
		elif self.softmax_dimension == 1:
			hot_vector = np.zeros([self._policy_size])
			hot_vector[action] = 1.0
			return hot_vector
		else:
			hot_vector = np.zeros([self._policy_size, self.softmax_dimension])
			for i in range(self._policy_size):
				hot_vector[i][action[i]] = 1.0
			return hot_vector
		
	def __init__( self, id, state_shape, policy_size, entropy_beta, learning_rate_input, device, concat_size = 0, softmax_dim = 1 ):
		# learning rate stuff
		self.train_count = 0
		self.entropy_beta = entropy_beta
		self.learning_rate_input = learning_rate_input
		# initialize
		self._id = id # model id
		self._device = device # gpu or cpu
		self._policy_size = int(policy_size) # the dimension of the policy vector
		self.softmax_dimension = softmax_dim # dimension of the softmax: 0 if none, 1 if you want a single softmax for the whole policy, 2 or more if you want a softmax (with dimension self.softmax_dimension) for any policy element
		self._concat_size = concat_size # the size of the vector concatenated with the CNN output before entering the LSTM
		self._state_shape = state_shape # the shape of the input
		self._lstm_units = 128 # the number of units of the LSTM
		self._create_network()
	
	def _create_network(self):
		scope_name = "net_{0}".format(self._id) # specify a tensorflow scope
		with tf.device(self._device), tf.variable_scope(scope_name, reuse=False) as scope:
			# [Input]
			self._input = tf.placeholder(tf.float32, np.concatenate([[None], self._state_shape], 0)) # the input placeholder -> "None" stands for the unknown batch size
			if self._concat_size > 0:
				self._concat = tf.placeholder(tf.float32, [None, self._concat_size]) # the concat placeholder -> "None" stands for the unknown batch size
			# [CNN tower]
			tower = self._create_tower_layer(self._input) # build the tower layer
			print( "Tower {0} shape: {1}".format(self._id, tower.get_shape()) )
			# [LSTM]
			lstm, self._lstm_state = self._create_lstm_layer(tower) # build the LSTM layer
			print( "LSTM {0} shape: {1}".format(self._id, lstm.get_shape()) )
			# [Policy]
			self._policy = self._create_policy_layer(lstm) # build the policy layer
			print( "Policy {0} shape: {1}".format(self._id, self._policy.get_shape()) )
			# [Value]
			self._value	= self._create_value_layer(lstm) # build the value layer
			print( "Value {0} shape: {1}".format(self._id, self._value.get_shape()) )
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name) # all the trainable variables
		
	def _create_tower_layer(self, input):
		input = tf.layers.conv2d( inputs=input, filters=16, kernel_size=(1,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
		# input = tf.layers.conv2d( inputs=input, filters=32, kernel_size=(3,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )
		return input
		
	def _create_lstm_layer(self, input):
		self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self._lstm_units, forget_bias=1.0, state_is_tuple=True)
		self._initial_lstm_state0 = tf.placeholder(tf.float32, [1, self._lstm_units])
		self._initial_lstm_state1 = tf.placeholder(tf.float32, [1, self._lstm_units])
		self._initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self._initial_lstm_state0, self._initial_lstm_state1)
		step_size = tf.shape(input)[:1] # input shape: (batch,w,h,depth)
		
		input = tf.layers.flatten(input) # input shape: (batch,w*h*depth)
		input = tf.layers.dense( inputs=input, units=2*self._lstm_units, activation=None, kernel_initializer=tf.initializers.variance_scaling )
		
		input = tf.contrib.layers.maxout( inputs=input, num_units=self._lstm_units, axis=-1 )
		input = tf.reshape(input, [-1, self._lstm_units]) # input shape: (batch,self._lstm_units)
		if self._concat_size > 0:
			input = tf.concat([input, self._concat], -1) # input shape: (batch,self._lstm_units+self._concat_size)
		
		input = tf.expand_dims(input, 0) # input shape: (1,batch,self._lstm_units+self._concat_size) -> at least 3 dimensions are required by the LSTM, the last dimension must be known
		lstm, lstm_state = tf.nn.dynamic_rnn(self.lstm_cell, input, initial_state = self._initial_lstm_state, sequence_length = step_size, time_major = False)
		return lstm, lstm_state
		
	def _create_policy_layer(self, input):
		if self.softmax_dimension < 2:
			policy = tf.layers.dense( inputs=input, units=self._policy_size, activation=None, kernel_initializer=tf.initializers.variance_scaling )
			policy = tf.reshape(policy, [-1, self._policy_size])
			if self.softmax_dimension == 1:
				return tf.nn.softmax(policy)
			else:
				return policy
		else:
			policy = []
			for i in range(self.softmax_dimension):
				p = tf.layers.dense( inputs=input, units=self._policy_size, activation=None, kernel_initializer=tf.initializers.variance_scaling )
				p = tf.reshape(p, [-1, self._policy_size])
				p = tf.expand_dims(p, axis=-1)
				policy.append(p)
			return tf.contrib.layers.softmax(tf.concat(policy, -1))

	def _create_value_layer(self, input, reuse=False):
		# Value (output)
		input = tf.layers.dense( inputs=input, units=1, activation=None, kernel_initializer=tf.initializers.variance_scaling )
		return tf.reshape(input, [-1]) # flatten output

	def _base_loss(self): # the loss function
		if self.softmax_dimension < 2:
			self.action_batch = tf.placeholder(tf.float32, [None, self._policy_size]) # "None" stands for the unknown batch size
		else:
			self.action_batch = tf.placeholder(tf.float32, [None, self._policy_size, self.softmax_dimension]) # "None" stands for the unknown batch size
		
		# Advantage (R-V) (input for policy)
		self.advantage_batch = tf.placeholder(tf.float32, [None]) # "None" stands for the unknown batch size
		
		# Avoid NaN with clipping when value in pi becomes zero
		log_policy = tf.log(tf.clip_by_value(self._policy, 1e-20, 1.0))
		
		reduction_indices=1
		if self.softmax_dimension >= 2:
			reduction_indices = [1,2]
		
		# Policy entropy
		entropy = -tf.reduce_sum(self._policy * log_policy, reduction_indices=reduction_indices)
		
		# Policy loss (output)
		policy_loss = -tf.reduce_sum( tf.reduce_sum( tf.multiply( log_policy, self.action_batch ), reduction_indices=reduction_indices ) * self.advantage_batch + entropy * self.entropy_beta )
		
		# R (input for value target)
		self.reward_batch = tf.placeholder(tf.float32, [None]) # "None" stands for the unknown batch size
		
		# Value loss (output)
		# (Learning rate for Critic is half of Actor's, so multiply by 0.5)
		value_loss = 0.5 * tf.nn.l2_loss(self.reward_batch - self._value)
		
		base_loss = policy_loss + value_loss
		return base_loss

	def prepare_loss(self):
		with tf.device(self._device):
			self.total_loss = self._base_loss()

	def reset_LSTM_state(self):
		self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, self._lstm_units]), np.zeros([1, self._lstm_units]))

	def run_policy_and_value(self, sess, state, concat = None):
		# This run_policy_and_value() is used when forward propagating.
		# so the step size is 1.
		feed_dict = { 
				self._input : state, 
				self._initial_lstm_state0 : self.lstm_state_out[0], 
				self._initial_lstm_state1 : self.lstm_state_out[1]
			}
		if self._concat_size > 0:
			feed_dict.update( { self._concat : concat } )
		pi_out, v_out, self.lstm_state_out = sess.run( [self._policy, self._value, self._lstm_state], feed_dict = feed_dict )
		# print(pi_out.shape)
		return pi_out.sum(axis=0)/pi_out.shape[0], v_out.sum(axis=0)/v_out.shape[0]
		
	def run_value(self, sess, state, concat = None):
		# This run_value() is used for calculating V for bootstrapping at the 
		# end of LOCAL_T_MAX time step sequence.
		# When next sequence starts, V will be calculated again with the same state using updated network weights,
		# so we don't update LSTM state here.
		feed_dict = {
				self._input : state, 
				self._initial_lstm_state0 : self.lstm_state_out[0], 
				self._initial_lstm_state1 : self.lstm_state_out[1] 
			}
		if self._concat_size > 0:
			feed_dict.update( { self._concat : concat } )
		v_out, _ = sess.run( [self._value, self._lstm_state], feed_dict = feed_dict )
		return v_out.sum(axis=0)/v_out.shape[0]
		
	def train(self, sess, gradient, learning_rate, states, actions, advantages, cumulative_rewards, start_lstm_state, concat = None):
		feed_dict={ 
					self.learning_rate_input: learning_rate,
					self._input: states,
					self.action_batch: actions,
					self.advantage_batch: np.reshape(advantages,[-1]),
					self.reward_batch: np.reshape(cumulative_rewards,[-1]),
					self._initial_lstm_state: start_lstm_state }
		if self._concat_size > 0:
			feed_dict.update( { self._concat : concat } )
		# Calculate gradients and copy them to global network.
		self.train_count += len(states)
		sess.run( gradient, feed_dict = feed_dict )

	def sync_from(self, src_network, name=None): # used for loading from checkpoints
		src_vars = src_network.get_vars()
		dst_vars = self.get_vars()
		sync_ops = []
		with tf.device(self._device):
			with tf.name_scope(name, "A3CModel{0}".format(self._id),[]) as name:
				for(src_var, dst_var) in zip(src_vars, dst_vars):
					sync_op = tf.assign(dst_var, src_var)
					sync_ops.append(sync_op)

				return tf.group(*sync_ops, name=name)