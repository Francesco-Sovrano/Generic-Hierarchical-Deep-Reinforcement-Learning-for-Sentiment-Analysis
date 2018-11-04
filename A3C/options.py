# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def build():
	# Common
	tf.app.flags.DEFINE_boolean("use_gpu", False, "whether to use the GPU")
	tf.app.flags.DEFINE_string("env_type", "sentipolc", "environment type")
	tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint", "checkpoint directory")
	tf.app.flags.DEFINE_string("event_dir", "./events", "events directory")
	tf.app.flags.DEFINE_string("log_dir", "./log", "events directory")
	tf.app.flags.DEFINE_boolean("show_best_screenshots", True, "whether to save the best matches")
	tf.app.flags.DEFINE_boolean("show_all_screenshots", False, "whether to save all the matches")
	
	tf.app.flags.DEFINE_string("test_set_path", "./database/test_set_sentipolc16.csv", "test set")
	tf.app.flags.DEFINE_string("training_set_path", "./database/training_set_sentipolc16.csv", "training set")
	tf.app.flags.DEFINE_string("emoji_sentiment_lexicon", "./database/Emoji_Sentiment_Data_v1.0.csv", "emoji sentiment lexicon")
	tf.app.flags.DEFINE_string("preprocessed_dict", "./database/preprocessed", "vectorized training set")
	tf.app.flags.DEFINE_string("translated_lemma_tokens", "./database/translated_lemma_tokens", "cache of translated lemma tokens") # dictionary with translated lemma tokens
	tf.app.flags.DEFINE_string("lexeme_sentiment_dict", "./database/lexeme_sentiment_dict", "cache of lexeme_sentiment") # lexeme sentiment dictionary
	tf.app.flags.DEFINE_string("test_annotations", "./database/test_annotations", "cache of test_annotations")
	tf.app.flags.DEFINE_string("training_annotations", "./database/training_annotations", "cache of training_annotations")
	
	tf.app.flags.DEFINE_string("tagger_path", "./.env2/treetagger", "tagger path")
	tf.app.flags.DEFINE_string("nltk_data", './.env2/nltk_data', "nltk data")
	tf.app.flags.DEFINE_string("word2vec_path", './.env2/word2vec/cc.it.300.bin', "word2vec data")
	
	tf.app.flags.DEFINE_string("task", "subjective, opos, oneg, ironic, lpos, lneg", "choose a combination of: subjective, opos, oneg, ironic, lpos, lneg")
	tf.app.flags.DEFINE_string("granularity", "lemma", "lemma or token")
	tf.app.flags.DEFINE_integer("gram_size", 1, "number of tokens/lemma to process at each step")
	tf.app.flags.DEFINE_integer("match_count_for_evaluation", 200, "number of matches used for evaluation scores")
	tf.app.flags.DEFINE_integer("parallel_size", 8, "parallel thread size")
	tf.app.flags.DEFINE_integer("situation_count", 3, "number of partitions considered by the algorithm")
	
	# For training
	tf.app.flags.DEFINE_float("gamma", 0.99, "discount factor for rewards") # doesn't work: 0.75
	tf.app.flags.DEFINE_integer("local_t_max", 5, "repeat step size") # doesn't work: 10
	tf.app.flags.DEFINE_float("entropy_beta", 0.001, "entropy regularization constant")
	tf.app.flags.DEFINE_integer("max_time_step", 6*10**6, "max time steps")
	tf.app.flags.DEFINE_integer("save_interval_step", 10**4, "saving interval steps")
	tf.app.flags.DEFINE_float("rmsp_alpha", 0.99, "decay parameter for rmsprop")
	tf.app.flags.DEFINE_float("rmsp_epsilon", 0.1, "epsilon parameter for rmsprop")
	tf.app.flags.DEFINE_float("initial_alpha_low", 1e-4, "log_uniform low limit for learning rate")
	tf.app.flags.DEFINE_float("initial_alpha_high", 5e-3, "log_uniform high limit for learning rate")
	tf.app.flags.DEFINE_float("initial_alpha_log_rate", 0.5, "log_uniform interpolate rate for learning rate")
	tf.app.flags.DEFINE_float("grad_norm_clip", 40.0, "gradient norm clipping")
	
def get():
	return tf.app.flags.FLAGS