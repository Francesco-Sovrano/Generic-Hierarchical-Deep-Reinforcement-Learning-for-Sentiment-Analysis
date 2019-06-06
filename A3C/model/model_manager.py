# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.a3c_model import A3CModel

import options
flags=options.get()

class ModelManager(object):
	def __init__( self, id, environment, learning_rate_input, device ):
		self._id=id # model id
		self._device=device # cpu or gpu
		
		agents_count=environment.get_situation_count() # manager output size
		action_size=environment.get_action_size() # agent output size
		state_shape=environment.get_state_shape() # input size
		if agents_count > 1:
			self.model_size=agents_count+1 # need for 1 extra agent as manager
			self.has_manager=True
		else:
			self.model_size=1
			self.has_manager=False
		# create networks
		self._model_list=[]
		self._model_usage_matrix=np.zeros(agents_count)
		self._model_usage_list=[]
		# the manager
		if self.has_manager:
			self.manager=A3CModel(id=str(id)+"_0", state_shape=state_shape, policy_size=agents_count, entropy_beta=flags.entropy_beta, learning_rate_input=learning_rate_input, device=device, concat_size=action_size, softmax_dim=1)
			self._model_list.append ( self.manager )
		# the agents
		for i in range(agents_count):
			# each situational agent has a different entropy
			agent=A3CModel(id=str(id)+"_"+str(i+1), state_shape=state_shape, policy_size=action_size, entropy_beta=flags.entropy_beta*(i+1), learning_rate_input=learning_rate_input, device=device, concat_size=action_size, softmax_dim=2)
			self._model_list.append (agent)
			
	def initialize(self, global_network, grad_applier):
		apply_gradients=[]
		sync=[]
		for i in range(self.model_size):
			local_agent=self.get_model(i)
			global_agent=global_network.get_model(i)
			local_agent.prepare_loss()
			apply_gradients.append( grad_applier.minimize_local(local_agent.total_loss, global_agent.get_vars(), local_agent.get_vars()) )
			sync.append( local_agent.sync_from(global_agent) ) # for synching local network with global one
		return apply_gradients, sync

	def get_model(self, id):
		return self._model_list[id]
		
	def get_statistics(self):
		stats={}
		if self.has_manager:
			total_usage=0
			usage_matrix=np.zeros(self.model_size-1, dtype=np.uint16)
			for u in self._model_usage_list:
				usage_matrix[u] += 1
				total_usage += 1
			for i in range(self.model_size-1):
				stats["model_{0}".format(i)]=usage_matrix[i]/total_usage if total_usage != 0 else 0
		return stats
		
	def get_agentID_by_state(self, sess, state, concat=None):
		if self.has_manager:
			policy, value=self.manager.run_policy_and_value(sess, state, concat)
			id=np.random.choice(range(len(policy)), p=policy)
			self._model_usage_matrix[id] += 1
			self._model_usage_list.append(id)
			if len(self._model_usage_list) > flags.match_count_for_evaluation:
				del self._model_usage_list[0]
				
			id=id + 1 # the first agent is the manager
			return id, policy, value
		else:
			return 0, None, None
	
	def get_vars(self):
		vars=[]
		for agent in self._model_list:
			vars=set().union(agent.get_vars(),vars)
		return list(vars)
		
	def reset(self):
		for agent in self._model_list:
			agent.reset_LSTM_state() # reset LSTM state

	def init_train_count(self):
		for agent in self._model_list:
			agent.train_count=0