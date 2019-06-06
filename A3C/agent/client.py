# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np
import time
import collections

from environment.environment import Environment
from model.model_manager import ModelManager

# get command line args
import options
flags = options.get()

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000


class Worker(object):
	def __init__(self, thread_index, session, annotated_wordbag, global_network, device, initial_learning_rate=None, learning_rate_input=None, grad_applier=None, train=True):
		self.train = train
		self.thread_index = thread_index
		self.sess = session
		self.global_network = global_network
		self.environment = Environment.create_environment(flags.env_type, self.thread_index, annotated_wordbag, shuffle=self.train)
		self.device = device
	# build network
		if self.train:
			self.local_network = ModelManager(self.thread_index, self.environment, learning_rate_input, self.device)
			self.apply_gradients, self.sync = self.local_network.initialize( self.global_network, grad_applier )
			self.initial_learning_rate = initial_learning_rate
		else:
			self.local_network = self.global_network
		self.terminal = True
		self.local_t = 0
		self.prev_local_t = 0
	#logs
		if self.train:
			# main log directory
			if not os.path.exists(flags.log_dir):
				os.makedirs(flags.log_dir)
			# episode result
			self.result_log_dir = flags.log_dir + "/thread" + str(self.thread_index)
			if not os.path.exists(self.result_log_dir):
				os.makedirs(self.result_log_dir)
			# perfomance
			if not os.path.exists(flags.log_dir + "/performance"):
				os.makedirs(flags.log_dir + "/performance")
			formatter = logging.Formatter('%(asctime)s %(message)s')
			# info logger
			self.info_logger = logging.getLogger('info_' + str(thread_index))
			hdlr = logging.FileHandler(flags.log_dir + '/performance/info_' + str(thread_index) + '.log')
			hdlr.setFormatter(formatter)
			self.info_logger.addHandler(hdlr) 
			self.info_logger.setLevel(logging.DEBUG)
			# reward logger
			self.reward_logger = logging.getLogger('reward_' + str(thread_index))
			hdlr = logging.FileHandler(flags.log_dir + '/performance/reward_' + str(thread_index) + '.log')
			hdlr.setFormatter(formatter)
			self.reward_logger.addHandler(hdlr) 
			self.reward_logger.setLevel(logging.DEBUG)
			
			self.max_reward = float("-inf")
			self.update_statistics()
		
	def update_statistics(self):
		self.stats = self.environment.get_statistics()
		self.stats.update(self.local_network.get_statistics())

	def prepare(self, episode_id=None): # initialize a new episode
		self.terminal = False
		self.environment.reset(episode_id)
		self.local_network.reset()

	def stop(self): # stop current episode
		self.environment.stop()
		
	def _anneal_learning_rate(self, global_step): # anneal learning rate
		learning_rate = self.initial_learning_rate * (flags.max_time_step - global_step) / flags.max_time_step
		if learning_rate < 0.0:
			learning_rate = 0.0
		return learning_rate

	def set_start_time(self, start_time, reset):
		self.start_time = start_time
		if reset:
			self.local_network.init_train_count()

	def _print_log(self, step):
		# if self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
			# self.prev_local_t += PERFORMANCE_LOG_INTERVAL
			# elapsed_time = time.time() - self.start_time
			# steps_per_sec = step / elapsed_time
			# print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format( step, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
		# Print statistics
		self.reward_logger.info( str(["{0}={1}".format(key,value) for key, value in self.stats.items()]) )
		# Print match results
		print_result = False
		if flags.show_all_screenshots:
			print_result = True
		elif flags.show_best_screenshots:
			if self.environment.episode_reward > self.max_reward:
				self.max_reward = self.environment.episode_reward
				print_result = True
		if print_result: # show episodes insides
			file = open(self.result_log_dir + '/reward({0})_epoch({1})_step({2}).log'.format(self.environment.episode_reward, self.environment.epoch, step),"w")
			file.write( 'Annotation: {0}\n'.format( ["{0}={1}".format(key,value) for key, value in sorted(self.environment.annotation.items(), key=lambda t: t[0])] ))
			file.write( 'Prediction: {0}\n'.format( ["{0}={1}".format(key,value) for key, value in sorted(self.environment.get_labeled_prediction().items(), key=lambda t: t[0])] ))
			file.write( 'Reward: {0}\n'.format(self.environment.compute_reward()) )
			file.close()
				
	# run simulations and build a batch for training
	def _build_batch(self, step):
		batch = {}
		if self.train: # init batch
			batch["states"] = []
			batch["actions"] = []
			batch["concat"] = []
			batch["cumulative_rewards"] = []
			batch["advantages"] = []
			batch["start_lstm_state"] = []
			for i in range(self.local_network.model_size):
				for key in batch:
					batch[key].append(collections.deque())
				batch["start_lstm_state"][i] = self.local_network.get_model(i).lstm_state_out
			
			agent_id_list = collections.deque()
			agent_reward_list = collections.deque()
			agent_value_list = collections.deque()
			manager_value_list = collections.deque()
		
		t = 0
		while t < flags.local_t_max and not self.terminal:
			t += 1
			prediction = self.environment.sentidoc
			state = self.environment.get_state()
			
			agent_id, manager_policy, manager_value = self.local_network.get_agentID_by_state(sess=self.sess, state=[state], concat=[prediction])
			agent = self.local_network.get_model(agent_id)
			agent_policy, agent_value = agent.run_policy_and_value(sess=self.sess, state=[state], concat=[prediction])
			
			action = self.environment.choose_action(agent_policy)
			reward, self.terminal = self.environment.process_action(action)
			
			self.local_t += 1
			if self.train: # populate batch
				if self.terminal: # update statistics
					self.update_statistics() # required before assigning manager reward
				# Populate agent batch
				batch["states"][agent_id].append(state)
				batch["actions"][agent_id].append(agent.get_action_vector(action))
				batch["concat"][agent_id].append(prediction)
				agent_id_list.appendleft(agent_id) # insert on top
				agent_reward_list.appendleft(reward) # insert on top
				agent_value_list.appendleft(agent_value) # insert on top
				# Populate manager batch
				if self.local_network.has_manager:
					batch["states"][0].append(state)
					batch["actions"][0].append(self.local_network.manager.get_action_vector(agent_id-1))
					batch["concat"][0].append(prediction)
					manager_value_list.appendleft(manager_value) # insert on top
				if (self.local_t % LOG_INTERVAL == 0):
					self.info_logger.info( "actions={0} value={1} agent={2}".format(agent_policy, agent_value, agent_id) )
		# build cumulative reward
		if self.train: 
			cumulative_reward = 0.0
			# if the episode has not terminated, bootstrap the value from the last state
			if not self.terminal:
				prediction = self.environment.sentidoc
				state = self.environment.get_state()
				agent_id, manager_policy, manager_value = self.local_network.get_agentID_by_state(sess=self.sess, state=[state], concat=[prediction])
				agent = self.local_network.get_model(agent_id)
				agent_value = agent.run_value(self.sess, state=[state], concat=[prediction])
				if self.local_network.has_manager:
					cumulative_reward = manager_value if abs(manager_value) > abs(agent_value) else agent_value # should prevent value over-estimation
				else:
					cumulative_reward = agent_value
				
			if self.local_network.has_manager:
				for(agent_id, agent_reward, agent_value, manager_value) in zip(agent_id_list, agent_reward_list, agent_value_list, manager_value_list):
					value = manager_value if abs(manager_value) > abs(agent_value) else agent_value
					cumulative_reward = agent_reward + flags.gamma * cumulative_reward
					batch["cumulative_rewards"][agent_id].appendleft(cumulative_reward) # insert on top
					batch["advantages"][agent_id].appendleft(cumulative_reward - value) # insert on top
					batch["cumulative_rewards"][0].appendleft(cumulative_reward) # insert on top
					batch["advantages"][0].appendleft(cumulative_reward - value) # insert on top
			else:
				for(agent_id, agent_reward, agent_value) in zip(agent_id_list, agent_reward_list, agent_value_list):
					cumulative_reward = agent_reward + flags.gamma * cumulative_reward
					batch["cumulative_rewards"][agent_id].appendleft(cumulative_reward) # insert on top
					batch["advantages"][agent_id].appendleft(cumulative_reward - agent_value) # insert on top
		return batch
	
	def process(self, step=0):
		if self.terminal:
			self.prepare()
			
		start_local_t = self.local_t
		# Copy weights from shared to local
		if self.train:
			learning_rate = []
			agents_count = self.local_network.model_size-1
			for i in range(self.local_network.model_size):
				self.sess.run(self.sync[i])
				rate = self._anneal_learning_rate(self.local_network.get_model(i).train_count)
				# manager learning rate should be the highest
				if i>0:
					rate /= agents_count
				learning_rate.append(rate)

		# Build feed dictionary
		batch_dict = self._build_batch(step)
		# Pupulate the feed dictionary
		if self.train:
			for i in range(self.local_network.model_size):
				if len(batch_dict["states"][i]) > 0:
					agent = self.local_network.get_model(i)
					agent.train(self.sess, self.apply_gradients[i], learning_rate[i], batch_dict["states"][i], batch_dict["actions"][i], batch_dict["advantages"][i], batch_dict["cumulative_rewards"][i], batch_dict["start_lstm_state"][i], batch_dict["concat"][i])
			self._print_log(step)
		diff_local_t = self.local_t - start_local_t
		return diff_local_t # local steps amount