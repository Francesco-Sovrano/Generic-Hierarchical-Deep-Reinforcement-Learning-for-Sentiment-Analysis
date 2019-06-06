# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from multiprocessing import Queue
import threading
import signal
import math
import os
import logging
import time
import pickle
import copy
from datetime import datetime

from environment.environment import Environment
from model.model_manager import ModelManager
from model.rmsprop_applier import RMSPropApplier
from agent.client import Worker

import options
options.build()
flags = options.get()

class Application(object):
	def __init__(self):
		# Training logger
		self.training_logger = logging.getLogger('results')
		if not os.path.isdir(flags.log_dir):
			os.mkdir(flags.log_dir)
		hdlr = logging.FileHandler(flags.log_dir + '/results.log')
		formatter = logging.Formatter('%(asctime)s %(message)s')
		hdlr.setFormatter(formatter)
		self.training_logger.addHandler(hdlr) 
		self.training_logger.setLevel(logging.DEBUG)
		# Test logger
		self.test_logger = logging.getLogger('test')
		if not os.path.isdir(flags.log_dir):
			os.mkdir(flags.log_dir)
		hdlr = logging.FileHandler(flags.log_dir + '/test.log')
		# formatter = logging.Formatter('%(asctime)s %(message)s')
		# hdlr.setFormatter(formatter)
		self.test_logger.addHandler(hdlr) 
		self.test_logger.setLevel(logging.DEBUG)
		# Build training and test set
		self.training_set, self.test_set = self.get_set(flags.preprocessed_dict+'.pkl')
		# Shuffle training set (no need to shuffle test set)
		np.random.shuffle(self.training_set)
		# Initialize network
		self.device = "/cpu:0"
		if flags.use_gpu:
			self.device = "/gpu:0"
		config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True) # prepare session
		if flags.use_gpu:
			config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		self.global_t = 0
		self.pause_requested = False
		self.terminate_requested = False
		self.build_network()
		
	def get_set(self, path):
		with open(path, 'rb') as f:
			pkl = pickle.load(f)
			return pkl['training_set'], pkl['test_set']
			
	def build_network(self):
		learning_rate_input = tf.placeholder("float")
		grad_applier = self.build_global_network(learning_rate_input)
		self.build_local_networks(learning_rate_input, grad_applier)
		self.sess.run(tf.global_variables_initializer()) # do it before loading checkpoint
		self.load_checkpoint()
		
	def build_global_network(self, learning_rate_input):
		environment = Environment.create_environment(flags.env_type, -1, self.training_set, shuffle=False)
		self.global_network = ModelManager( -1, environment, learning_rate_input, self.device )
		# return gradient optimizer
		return RMSPropApplier(learning_rate = learning_rate_input, decay = flags.rmsp_alpha, momentum = 0.0, epsilon = flags.rmsp_epsilon, clip_norm = flags.grad_norm_clip, device = self.device)
		
	def build_local_networks(self, learning_rate_input, grad_applier):
		initial_learning_rate = self.log_uniform(flags.initial_alpha_low, flags.initial_alpha_high, flags.initial_alpha_log_rate)
		self.trainers = []
		for i in range(flags.parallel_size):
			trainer = Worker(i, self.sess, self.training_set, self.global_network, self.device, initial_learning_rate, learning_rate_input, grad_applier)
			self.trainers.append(trainer)
			
	def log_uniform(self, lo, hi, rate):
		log_lo = math.log(lo)
		log_hi = math.log(hi)
		v = log_lo * (1-rate) + log_hi * rate
		return math.exp(v)
			
	def train_function(self, parallel_index, reset):
		""" Train each environment. """	
		trainer = self.trainers[parallel_index]		
		# set start_time
		trainer.set_start_time(self.start_time, reset)
		print( 'Thread {0} started'.format(parallel_index) )
		while True:
			if self.pause_requested:
				break
			if parallel_index == len(self.trainers)-1:
				if self.global_t > flags.max_time_step:
					self.terminate_requested = True
				if self.global_t > self.next_save_steps or self.terminate_requested:
					self.save() # Save checkpoint
			if self.terminate_requested or self.global_t > flags.max_time_step:
				trainer.stop()
				break
	
			diff_global_t = trainer.process(step=self.global_t)
			self.global_t += diff_global_t
			
			# print global statistics
			if trainer.terminal:
				info = {}
				for t in self.trainers:
					for key in t.stats:
						if not info.get(key):
							info[key] = 0
						info[key] += t.stats[key]
				self.training_logger.info( str([key + "=" + str(value/len(self.trainers)) for key, value in sorted(info.items(), key=lambda t: t[0])]) ) # Print statistics
				
	def test_function(self, parallel_index, tester):
		lines = []
		for id in range(parallel_index,len(self.test_set),flags.parallel_size):
			tester.prepare(id)
			while not tester.terminal:
				tester.process()
			
			environment = tester.environment
			annotation = copy.deepcopy(environment.annotation)
			for i in range(len(environment.sentidoc)):
				key = environment.get_task_by_index(i)
				annotation[key] = environment.sentidoc[i]
			line = '{0},{1},{2},{3},{4},{5},{6},{7}\n'.format( annotation["id"], annotation["subjective"], annotation["opos"], annotation["oneg"], annotation["ironic"], annotation["lpos"], annotation["lneg"], annotation["topic"] )
			lines.append(line)
		return lines

	def test(self):
		result_file = flags.log_dir + '/evaluation_' + str(self.global_t) + '.csv'
		if os.path.exists(result_file):
			print('Test results already produced and evaluated for ' + result_file)
			return
		print('Start testing')
		threads = []
		result_queue = Queue()
		for i in range(flags.parallel_size): # parallel testing
			tester = Worker(-1-i, self.sess, self.test_set, self.global_network, self.device, train=False)
			thread = threading.Thread(target=lambda q, arg1, arg2: q.put(self.test_function(arg1,arg2)), args=(result_queue,i,tester))
			thread.start()
			threads.append(thread)
		time.sleep(5)
		for thread in threads: # wait for all threads to end
			thread.join()
		with open(result_file, "w", encoding="utf-8") as file: # write results to file
			while not result_queue.empty():
				result = result_queue.get()
				for line in result:
					file.write(line)
		print('End testing')
		print('Test results saved in ' + result_file)
		return self.evaluate(result_file)

	def train(self):
		# run training threads
		self.train_threads = []
		for i in range(flags.parallel_size):
			self.train_threads.append(threading.Thread(target=self.train_function, args=(i,True)))
		signal.signal(signal.SIGINT, self.signal_handler)
		# set start time
		self.start_time = time.time() - self.wall_t
		for t in self.train_threads:
			t.start()
		time.sleep(5)
		print('Press Ctrl+C to stop')
		signal.pause()
		
	def load_checkpoint(self):
		# init or load checkpoint with saver
		self.saver = tf.train.Saver(var_list=self.global_network.get_vars(), max_to_keep=0) # keep all checkpoints
		checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print("checkpoint loaded:", checkpoint.model_checkpoint_path)
			tokens = checkpoint.model_checkpoint_path.split("-")
			# set global step
			self.global_t = int(tokens[1])
			print(">>> global step set: ", self.global_t)
			# set wall time
			wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
			with open(wall_t_fname, 'r') as f:
				self.wall_t = float(f.read())
				self.next_save_steps = (self.global_t + flags.save_interval_step) // flags.save_interval_step * flags.save_interval_step
		else:
			print("Could not find old checkpoint")
			# set wall time
			self.wall_t = 0.0
			self.next_save_steps = flags.save_interval_step
			
	def save(self):
		""" Save checkpoint. 
		Called from thread-0.
		"""
		self.pause_requested = True
		for (i, t) in enumerate(self.train_threads): # Wait for all other threads to stop
			if i != len(self.train_threads)-1: # cannot join current thread
				t.join()
	
		# Save
		if not os.path.exists(flags.checkpoint_dir):
			os.mkdir(flags.checkpoint_dir)
	
		# Write wall time
		wall_t = time.time() - self.start_time
		wall_t_fname = flags.checkpoint_dir + '/wall_t.' + str(self.global_t)
		with open(wall_t_fname, 'w') as f:
			f.write(str(wall_t))
	
		print('Start saving')
		self.saver.save(self.sess, flags.checkpoint_dir + '/checkpoint', global_step = self.global_t)
		print('End saving')
		
		# Test
		test_result = self.test()
		
		# Restart training
		if not self.terminate_requested:
			self.pause_requested = False
			self.next_save_steps += flags.save_interval_step
			# Restart other threads
			for i in range(flags.parallel_size):
				if i != len(self.train_threads)-1: # current thread is already running
					thread = threading.Thread(target=self.train_function, args=(i,False))
					self.train_threads[i] = thread
					thread.start()
					
	def evaluate(self, result_file):
		print('Start evaluating')
		verbose=True
		self.test_logger.info(datetime.now())
		# read gold standard and populate the count matrix
		gold = dict()
		gold_counts =  {'subj':{'0':0,'1':0},
						'opos':{'0':0,'1':0},
						'oneg':{'0':0,'1':0},
						'iro':{'0':0,'1':0},
						'lpos':{'0':0,'1':0},
						'lneg':{'0':0,'1':0}
						}
		with open(flags.test_set_path) as f:
			for line in f:
				raw = line.rstrip().split(',')
				id = str(raw[0].replace('"', ''))
				subj = str(raw[1].replace('"', ''))
				opos = str(raw[2].replace('"', ''))
				oneg = str(raw[3].replace('"', ''))
				iro = str(raw[4].replace('"', ''))
				lpos = str(raw[5].replace('"', ''))
				lneg = str(raw[6].replace('"', ''))
				top = str(raw[7].replace('"', ''))
				
				#id, subj, opos, oneg, iro, lpos, lneg, top = map(lambda x: x[1:-1], line.rstrip().split(','))
				gold[id] = {'subj':subj, 'opos':opos, 'oneg':oneg, 'iro':iro, 'lpos':lpos, 'lneg':lneg}
				gold_counts['subj'][subj]+=1
				gold_counts['opos'][opos]+=1
				gold_counts['oneg'][oneg]+=1
				gold_counts['iro'][iro]+=1

				gold_counts['lpos'][lpos]+=1
				gold_counts['lneg'][lneg]+=1
				
		# read result data
		result = dict()
		with open(result_file) as f:
			for line in f:
				raw = line.rstrip().split(',')
				id = str(raw[0].replace('"', ''))
				subj = str(raw[1].replace('"', ''))
				opos = str(raw[2].replace('"', ''))
				oneg = str(raw[3].replace('"', ''))
				iro = str(raw[4].replace('"', ''))
				lpos = str(raw[5].replace('"', ''))
				lneg = str(raw[6].replace('"', ''))
				top = str(raw[7].replace('"', ''))
				result[id]= {'subj':subj, 'opos':opos, 'oneg':oneg, 'iro':iro}
				
		task_f1 = {}
		# evaluation: single classes
		for task in ['subj', 'opos', 'oneg', 'iro']:	#add 'lpos' and 'lneg' if you want to measure literal polairty
			# table header
			if verbose: self.test_logger.info ("\ntask: {}".format(task))
			if verbose: self.test_logger.info ("prec. 0\trec. 0\tF-sc. 0\tprec. 1\trec. 1\tF-sc. 1\tF-sc.")
			correct =  {'0':0,'1':0}
			assigned = {'0':0,'1':0}
			precision ={'0':0.0,'1':0.0}
			recall =   {'0':0.0,'1':0.0}
			fscore =   {'0':0.0,'1':0.0}
		   
			# count the labels
			for id, gold_labels in gold.items():
				if (not id in result) or result[id][task]=='':
					pass
				else:
					assigned[result[id][task]] += 1					
					if gold_labels[task]==result[id][task]:
						correct[result[id][task]] += 1

			# compute precision, recall and F-score
			for label in ['0','1']:
				try:
					precision[label] = float(correct[label])/float(assigned[label])
					recall[label] = float(correct[label])/float(gold_counts[task][label])
					fscore[label] = (2.0 * precision[label] * recall[label]) / (precision[label] + recall[label])
				except:
					# if a team doesn't participate in a task it gets default 0 F-score
					fscore[label] = 0.0
					
			task_f1[task] = (fscore['0'] + fscore['1'])/2.0
			# write down the table
			self.test_logger.info("{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}".format( 
					precision['0'], recall['0'], fscore['0'], 
					precision['1'], recall['1'], fscore['1'],
					task_f1[task]))
										
		# polarity evaluation needs a further step
		if verbose: self.test_logger.info("\ntask: polarity")
		if verbose: self.test_logger.info("Combined F-score")
		correct =  {'opos':{'0':0,'1':0}, 'oneg':{'0':0,'1':0}}
		assigned = {'opos':{'0':0,'1':0}, 'oneg':{'0':0,'1':0}}
		precision ={'opos':{'0':0.0,'1':0.0}, 'oneg':{'0':0.0,'1':0.0}}
		recall =   {'opos':{'0':0.0,'1':0.0}, 'oneg':{'0':0.0,'1':0.0}}
		fscore =   {'opos':{'0':0.0,'1':0.0}, 'oneg':{'0':0.0,'1':0.0}}

		# count the labels
		for id, gold_labels in gold.items():
			for cl in ['opos','oneg']:
				if (not id in result) or result[id][cl]=='':
					pass
				else:
					assigned[cl][result[id][cl]] += 1
					if gold_labels[cl]==result[id][cl]:
						correct[cl][result[id][cl]] += 1
					
		# compute precision, recall and F-score
		for cl in ['opos','oneg']:
			for label in ['0','1']:
				try:
					precision[cl][label] = float(correct[cl][label])/float(assigned[cl][label])
					recall[cl][label] = float(correct[cl][label])/float(gold_counts[cl][label])
					fscore[cl][label] = float(2.0 * precision[cl][label] * recall[cl][label]) / float(precision[cl][label] + recall[cl][label])
				except:
					fscore[cl][label] = 0.0

		fscore_pos = (fscore['opos']['0'] + fscore['opos']['1'] ) / 2.0
		fscore_neg = (fscore['oneg']['0'] + fscore['oneg']['1'] ) / 2.0

		# write down the table
		task_f1["polarity"] = (fscore_pos + fscore_neg)/2.0
		self.test_logger.info("{0:.4f}".format(task_f1["polarity"]))
		print('End evaluating')
		return task_f1
		
	def signal_handler(self, signal, frame):
		print('You pressed Ctrl+C!')
		self.terminate_requested = True