# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from environment import environment

# get command line args
import options
flags = options.get()

import numpy as np
import os
import math

class SentiPolcEnvironment(environment.Environment):
	
	def get_situation_count(self):
		return flags.situation_count
		
	def get_state_shape(self):
		return ( flags.gram_size, 300+6, 1 )
		
	def get_state(self):
		state = np.zeros( self.get_state_shape() )
		for i in range(flags.gram_size):
			word_index = self.step + i
			if word_index < self.max_step-1:
				token = self.tokens[word_index]
				v = np.concatenate( (token[flags.granularity+"_vector"]*100,self.get_word_meta(token)) )
				state[i] = np.expand_dims(v, axis=-1)
			elif word_index == self.max_step-1:
				v = np.concatenate( (self.docvec*100,self.get_word_meta(None)) )
				state[i] = np.expand_dims(v, axis=-1)
		return state
		
	def get_word_meta(self, token):
		if token is None:
			return [0, 0, 0, 0, 0, 16] # is docvec
		lexeme_sentiment = token["lexeme_sentiment"]
		negativity = 0
		positivity = 0
		is_emoji = 0
		if len(lexeme_sentiment)>0:
			negativity = lexeme_sentiment["0"]["negativity"]
			positivity = lexeme_sentiment["0"]["positivity"]
			if "shortcode" in lexeme_sentiment["0"]:
				is_emoji = 1
		return [negativity, positivity, token["is_negator"]*2, token["is_intensifier"]*4, is_emoji*8, 0]
				
	def compute_reward(self):
		reward = 0
		confusion = self.get_confusion_matrix()
		for i in range(self.get_action_size()):
			key = self.get_task_by_index(i)
			tp = confusion[key][0][0]
			fp = confusion[key][0][1]
			tn = confusion[key][1][1]
			fn = confusion[key][1][0]
			if not self.terminal: # doesn't work without it
				if tp:
					reward += 0.05
				elif fn:
					reward += -0.05
			else:
				if tp:
					reward += 1
				elif tn:
					reward += 0.1 # doesn't work without it
				elif fp:
					reward += -1
				elif fn:
					reward += -0.1
		return reward
		
	def __init__(self, thread_index, documents, shuffle):
		self.task_id = [task.strip() for task in flags.task.split(',')]
		self.reward_list = []
		self.confusion_list = []
		self.thread_index = thread_index
		self.shuffle = shuffle
		self.documents = documents # passage by reference
		self.epoch = 0
		self.id = self.thread_index
		
	def reset(self, id=None):
		if id is None:
			self.id += flags.parallel_size
			if self.id >= len(self.documents): # start new epoch
				if self.shuffle and self.thread_index==0:
					np.random.shuffle(self.documents)
				self.id = self.thread_index
				self.epoch += 1
		else:
			self.id = id
		# retrieve annotated wordbag
		doc = self.documents[self.id]
		self.tokens = self.remove_noisy_tokens( doc["tokens_annotation"] ) # no stopwords
		self.annotation = doc["text_annotation"]
		self.docvec = doc["average_docvec"]
		# init episode
		self.sentidoc = np.zeros(self.get_action_size(), dtype=np.uint8)
		self.terminal = False
		self.episode_reward = 0
		self.step = 0
		self.max_step = len(self.tokens)+1 # words + context (docvec)
		
	def remove_noisy_tokens(self, tokens):
		new_tokens = []
		for token in tokens:
			if token["is_stop"]==0 and token["is_uri"]==0:
				new_tokens.append(token)
		return new_tokens
		
	def process_action(self, action):
		if self.step+flags.gram_size >= self.max_step:
			self.terminal = True # do it before computing reward
		self.old_prediction = self.sentidoc
		self.sentidoc = action
		
		if self.terminal: # Confusion
			self.confusion_list.append( self.get_confusion_matrix() )
			if len(self.confusion_list) > flags.match_count_for_evaluation:
				del self.confusion_list[0]
		
		reward = self.compute_reward()
		self.episode_reward += reward
		
		if self.terminal: # Reward
			self.reward_list.append(self.episode_reward)
			if len(self.reward_list) > flags.match_count_for_evaluation:
				del self.reward_list[0]
		
		self.step += flags.gram_size # it changes the current state, do it as last command of this function (otherwise error!!)
		return reward, self.terminal
	
	def choose_action(self, policy):
		action = []
		shape = np.shape(policy)
		for i in range(shape[0]):
			action.append(np.random.choice(range(shape[1]), p=policy[i]))
		return action
		
	def get_action_size(self):
		return len(self.task_id)
	
	def stop(self):
		pass
	
	def get_task_by_index(self, index):
		return self.task_id[index]
	
	def get_labeled_prediction(self):
		dict = {}
		for i in range(self.get_action_size()):
			key = self.get_task_by_index(i)
			value = self.sentidoc[i]
			dict[key] = value
		return dict
		
	def get_confusion_matrix(self):
		confusion = {}
		for i in range(self.get_action_size()):
			key = self.get_task_by_index(i)
			value = self.sentidoc[i]
			confusion[key] = np.zeros((2,2))
			# Win
			if self.annotation[key] == value:
				if value == 1: # true positive
					confusion[key][0][0] = 1
				else: # true negative
					confusion[key][1][1] = 1
			# Lose
			else:
				if value == 1: # false positive
					confusion[key][0][1] = 1
				else: # false negative
					confusion[key][1][0] = 1
		return confusion
		
	def get_statistics(self, start=None):
		if start is None:
			start = 0
		tp = {} # true positive
		tn = {} # true negative
		fp = {} # false positive
		fn = {} # false negative
		for i in range(self.get_action_size()):
			key = self.get_task_by_index(i)
			tp[key]=0
			tn[key]=0
			fp[key]=0
			fn[key]=0
		for confusion in self.confusion_list[start:]:
			for key, value in confusion.items():
				tp[key] += value[0][0]
				tn[key] += value[1][1]
				fp[key] += value[0][1]
				fn[key] += value[1][0]
		stats = {}
		stats["avg_reward"] = sum(self.reward_list)/len(self.reward_list) if len(self.reward_list) != 0 else 0
		stats["epoch"] = self.epoch
		stats["avg_mcc"] = 0
		for i in range(self.get_action_size()):
			key = self.get_task_by_index(i)
			# stats[key+"_p+"], stats[key+"_r+"], stats[key+"_f1+"] = self.get_positive_fscore(tp[key], tn[key], fp[key], fn[key])
			# stats[key+"_p-"], stats[key+"_r-"], stats[key+"_f1-"] = self.get_negative_fscore(tp[key], tn[key], fp[key], fn[key])
			_, _, stats[key+"_f1+"] = self.get_positive_fscore(tp[key], tn[key], fp[key], fn[key])
			_, _, stats[key+"_f1-"] = self.get_negative_fscore(tp[key], tn[key], fp[key], fn[key])
			# stats[key+"_precision"] = (stats[key+"_p+"]+stats[key+"_p-"])/2
			# stats[key+"_recall"] = (stats[key+"_r+"]+stats[key+"_r-"])/2
			stats[key+"_f1"] = (stats[key+"_f1+"]+stats[key+"_f1-"])/2
			# stats[key+"_accuracy"] = self.get_accuracy(tp[key], tn[key], fp[key], fn[key])
			stats[key+"_mcc"] = self.get_mcc(tp[key], tn[key], fp[key], fn[key])
			stats["avg_mcc"] += stats[key+"_mcc"]
		stats["avg_mcc"] /= self.get_action_size()
		return stats
		
	def get_positive_fscore(self, tp, tn, fp, fn):
		precision = tp/(tp+fp) if tp+fp != 0 else 0
		recall = tp/(tp+fn) if tp+fn != 0 else 0
		f1 = 2 * ((precision*recall)/(precision+recall)) if precision+recall != 0 else 0
		return precision, recall, f1
		
	def get_negative_fscore(self, tp, tn, fp, fn):
		precision = tn/(tn+fn) if tn+fn != 0 else 0
		recall = tn/(tn+fp) if tn+fp != 0 else 0
		f1 = 2 * ((precision*recall)/(precision+recall)) if precision+recall != 0 else 0
		return precision, recall, f1
		
	def get_mcc(self, tp, tn, fp, fn): # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
		denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
		if denominator != 0:
			return (tp*tn - fp*fn)/denominator
		return 2*self.get_accuracy(tp, tn, fp, fn)-1
	
	def get_accuracy(self, tp, tn, fp, fn):
		accuracy = (tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn != 0 else 0
		return accuracy
