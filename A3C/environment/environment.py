# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class Environment(object):
	# cached action size
	action_size = -1
	
	@staticmethod
	def create_environment(env_type, thread_index, annotated_doc2vec, shuffle):
		from . import sentipolc_environment
		return sentipolc_environment.SentiPolcEnvironment(thread_index, annotated_doc2vec, shuffle)
			
	def print_display(self):
		pass

	def __init__(self):
		pass

	def process(self, action):
		pass

	def reset(self):
		pass

	def stop(self):
		pass
