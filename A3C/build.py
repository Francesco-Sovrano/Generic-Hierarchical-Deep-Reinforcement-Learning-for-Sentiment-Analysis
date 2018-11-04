# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import time
import random
import math
import numpy as np
import string
import urllib.parse as urlparse
from validate_email import validate_email
import os
import pickle
import gc

import options
options.build()
flags = options.get()

# emoji
import unicodedata
from emojipy import Emoji
# treetagger
import treetaggerwrapper
# nltk
import nltk
nltk.download(['punkt','stopwords','sentiwordnet','wordnet','perluniprops','nonbreaking_prefixes'],flags.nltk_data)
from googletrans import Translator
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.moses import MosesTokenizer
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
# opener
from VUSentimentLexicon import LexiconMod as lm
# fasttext
import fastText as ft
# gensim
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

class SentiPolC(object):
	def build_set(self):
		wn.ensure_loaded()  # `LazyCorpusLoader` conversion into `WordNetCorpusReader` starts
		print ("WordNet loaded")
		swn.ensure_loaded()  # `LazyCorpusLoader` conversion into `SentiWordNetCorpusReader` starts
		print ("SentiWordNet loaded")
		self.tweet_tokenizer = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
		print ("Tweet tokenizer loaded")
		self.it_tokenizer = MosesTokenizer(lang='it')
		print ("Moses tokenizer loaded")
		self.it_tagger = treetaggerwrapper.TreeTagger(TAGLANG="it", TAGDIR=flags.tagger_path)
		# self.en_tagger = treetaggerwrapper.TreeTagger(TAGLANG="en", TAGDIR=flags.tagger_path)
		print ("Tagger loaded")
		self.stop_words = set(stopwords.words('italian'))
		print ("Stopwords loaded")
		self.lexicon = lm.LexiconSent('it')
		print ("OpeNER lexicon loaded")
		self.emoji = self.get_emoji_sentiment_lexicon(flags.emoji_sentiment_lexicon)
		print ("Emoji sentiment lexicon loaded")
		self.translator = Translator()
		print ("Setting up support dictionaries")
		self.translated_lemma_tokens = self.load_obj(flags.translated_lemma_tokens)
		self.lexeme_sentiment_dict = self.load_obj(flags.lexeme_sentiment_dict)
		print ("Translator loaded")
		# Build test annotations
		print ("Building test annotations..")
		test_set = self.load_obj(flags.test_annotations)
		if not test_set:
			test_set = self.get_annotations(flags.test_set_path)
			self.save_obj(test_set, flags.test_annotations)
		print ("Test annotations built")
		# Build training annotations
		print ("Building training annotations..")
		training_set = self.load_obj(flags.training_annotations)
		if not training_set:
			training_set = self.get_annotations(flags.training_set_path)
			self.save_obj(training_set, flags.training_annotations)
		print ("Training annotations built")
		print ("Saving support dictionaries")
		self.save_obj(self.translated_lemma_tokens, flags.translated_lemma_tokens)
		self.save_obj(self.lexeme_sentiment_dict, flags.lexeme_sentiment_dict)
		# Build distributional docvec from training and test sets
		self.doc2vec = self.build_distributional_docvec([test_set, training_set])
		print ("Doc2Vec built")
		self.add_context_to_annotations(test_set)
		print ("Distr. docvec added to test annotations")
		self.add_context_to_annotations(training_set)
		print ("Distr. docvec added to training annotations")
		self.free_ram()
		print ("Loading pre-trained model..")
		self.model = ft.load_model(flags.word2vec_path)
		print ("Pre-trained model loaded")
		self.add_wordvecs_to_annotations(test_set)
		print ("Wordvecs added to test annotations")
		self.add_wordvecs_to_annotations(training_set)
		print ("Wordvecs added to training annotations")
		# Save to npy
		self.free_ram()
		self.save_obj({"test_set":test_set, "training_set":training_set}, flags.preprocessed_dict)
		
	def free_ram(self):
		self.tweet_tokenizer = None
		self.it_tokenizer = None
		self.model = None
		self.it_tagger = None
		self.stop_words = None
		self.lexicon = None
		self.emoji = None
		self.translator = None
		self.doc2vec = None
		self.translated_lemma_tokens = None
		gc.collect()
		
	def save_obj(self, obj, path):
		path += '.pkl'
		print ("Saving " + path)
		with open(path, 'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
		print (path + " saved")
		
	def load_obj(self, path):
		path += '.pkl'
		if not os.path.isfile(path):
			return {}
		print ("Loading " + path)
		with open(path, 'rb') as f:
			return pickle.load(f)
		print (path + " loaded")
		
	# build a distributional polarity lexicon for emojis using http://kt.ijs.si/data/Emoji_sentiment_ranking/
	def get_emoji_sentiment_lexicon(self, path):
		emoji_sentiment_lexicon = {}
		with open(path, encoding="utf-8") as f:
			# a line is made of: (Emoji,Unicode codepoint,Occurrences,Position,Negative,Neutral,Positive,Unicode name,Unicode block)
			for line in f:
				raw = line.rstrip().split(',')
				negative = float(raw[4])
				neutral = float(raw[5])
				positive = float(raw[6])
				tot = negative+neutral+positive
				emoji_sentiment_lexicon.update({
					raw[0]: {
						"codepoint": raw[1],
						"occurences": int(raw[2]),
						"position": float(raw[3]),
						"negativity": negative/tot,
						"neutrality": neutral/tot,
						"positivity": positive/tot,
						"description": str(raw[7]),
						"block": str(raw[8]),
					}
				})
		return emoji_sentiment_lexicon
		
	# store in memory the dataset and its annotations
	def get_annotations(self, path):
		annotations = []
		with open(path, encoding="utf-8") as f:
			for line in f:
				raw = line.rstrip().split(',')
				id = str(raw[0].replace('"', ''))
				subj = int(raw[1].replace('"', ''))
				opos = int(raw[2].replace('"', ''))
				oneg = int(raw[3].replace('"', ''))
				iro = int(raw[4].replace('"', ''))
				lpos = int(raw[5].replace('"', ''))
				lneg = int(raw[6].replace('"', ''))
				topic = int(raw[7].replace('"', ''))
				text = str(raw[8])
				for i in range(9, len(raw)):
					text += ", " + str(raw[i])
				annotations.append( {
					"text": text,
					"text_annotation": { "id":id, "subjective": subj, "opos": opos, "oneg": oneg, "ironic": iro, "lpos": lpos, "lneg": lneg, "topic": topic },
				} )
		for a in annotations:
			tokens = self.tokenize(a["text"])
			a["tokens"] = tokens
			a["tokens_annotation"] = self.get_tokens_annotation(tokens)
		return annotations
		
	def get_tokens_annotation(self, tokens):
		# pos-tagging
		tags = treetaggerwrapper.make_tags(self.it_tagger.tag_text(tokens, tagonly=True)) # it doesn't use the TreeTagger's tokenization
		tokens_annotation = []
		for tag, token in zip(tags, tokens):
			pos, lemma = self.get_pos_lemma(tag, token)
		# Stop word
			stop = 1 if lemma in self.stop_words else 0
		# URI
			uri = 1 if self.is_url(lemma) or self.is_email(lemma) else 0
		# Interesting word
			interesting = 1 if self.is_interesting(pos) else 0
		# Special word
			special = 0 if self.starts_with_letter(lemma) else 1
		# Get sentiment from english SentiWordNet
			lexeme_sentiment = self.get_lexeme_sentiment(lemma, pos, stop, uri, interesting, special)
		# Get lemma properties from OpenER lexicon
			# polarity_tuple = self.lexicon.getPolarity(lemma,self.get_OpeNER_pos(pos))
			# polarity = 0
			# if polarity_tuple[0] == 'positive':
				# polarity = 1
			# elif polarity_tuple[0] == 'negative':
				# polarity = -1
			negator = 1 if self.lexicon.isNegator(lemma) else 0
			intensifier = 1 if self.lexicon.isIntensifier(lemma) else 0
			
			tokens_annotation.append( {
				"token": token,
				"lemma": lemma,
				"pos_tag": pos,
				# "opener_polarity": polarity,
				"lexeme_sentiment": lexeme_sentiment,
				"is_negator": negator,
				"is_intensifier": intensifier,
				"is_stop": stop,
				"is_special": special,
				"is_interesting": interesting,
				"is_uri": uri,
			} )
		return tokens_annotation		
	
	def get_lexeme_sentiment(self, lemma, pos, stop, uri, interesting, special):
		if stop==0 and uri==0:
			if interesting==1 and special==0:
				return self.get_SentiWordNet_sentiment(lemma, pos)
			elif special==1:
				return self.get_Emoji_sentiment(lemma)
		return {}
				
	def get_Emoji_sentiment(self, lemma):
		lexeme_sentiment = {}
		unicode_emoji = Emoji.ascii_to_unicode(lemma.upper())
		if len(unicode_emoji) == 1:
			category = unicodedata.category(unicode_emoji)
			if category == 'So': # is "Symbol other" (a specific unicode category)
				shortcode = Emoji.unicode_to_shortcode(unicode_emoji)
				if shortcode != unicode_emoji: # is an emoji only if it has an emoji shortcode
					# check whether the emoji is in the emoji polarity lexicon
					if unicode_emoji in self.emoji: # is in the lexicon
						emodict = self.emoji[unicode_emoji]
						lexeme_sentiment["0"] = { "shortcode":shortcode, "negativity":emodict["negativity"], "positivity":emodict["positivity"] }
					else: # tokenize the shortcode and get its tokens polarities from SentiWordNet
						tokens = shortcode.strip(' :').split('_') # shortcode tokenization seems an easy problem to solve
						negativity = 0
						positivity = 0
						count = 0 # count the number of shortcode tokens with a synset
						for token in tokens:
							synsets = list(swn.senti_synsets(token))
							if len(synsets)>0:
								senti_synset = synsets[0]
								negativity += senti_synset.neg_score()
								positivity += senti_synset.pos_score()
								count += 1
						if count > 1: # take the average of all shortcode tokens polarities
							negativity /= count
							positivity /= count
						lexeme_sentiment["0"] = { "shortcode":shortcode, "negativity":negativity, "positivity":positivity }
					# print(lexeme_sentiment["0"])
		return lexeme_sentiment
		
	def get_SentiWordNet_sentiment(self, lemma, pos):
		if lemma not in self.translated_lemma_tokens:
			en_lemma = ""
			while en_lemma=="": # workaround to handle google translator limitations
				try:
					en_lemma = self.translator.translate(lemma, dest='en', src='it').text
				except:
					traceback.print_exc()
					self.translator = Translator() # reset translator
					# time.sleep(random.uniform(0.01, 0.1))
					time.sleep(random.uniform(0.5, 1.5))
			en_lemma_tokens = self.tweet_tokenizer.tokenize(en_lemma)
			self.translated_lemma_tokens[lemma] = en_lemma_tokens
		else:
			en_lemma_tokens = self.translated_lemma_tokens[lemma]
		# print("EN: {}".format(en_lemma_tokens))
		wordnet_pos = self.get_WordNet_pos(pos)
		lex_key = "{}_{}".format(lemma,wordnet_pos)
		if lex_key not in self.lexeme_sentiment_dict:
			lexeme_sentiment = {}
			for en_token in en_lemma_tokens:
				if en_token:
					en_synsets = wn.synsets(en_token, pos=wordnet_pos)
					id = 0
					for syn in en_synsets:
						name = syn.name()
						senti_synset = swn.senti_synset(name) # this is a naive solution -> do you want to improve it? you need for something like Lesk algorithm for word disambiguation
						negativity = senti_synset.neg_score()
						positivity = senti_synset.pos_score()
						lexeme_sentiment[str(id)] = { "synset":name, "negativity":negativity, "positivity":positivity }
						id+=1
			self.lexeme_sentiment_dict[lex_key] = lexeme_sentiment
		else:
			lexeme_sentiment = self.lexeme_sentiment_dict[lex_key]
		return lexeme_sentiment
		
	def tokenize(self, text):
		tweet_tokens = self.tweet_tokenizer.tokenize(text) # tweet tokenisation
		tokens = []
		for tt in tweet_tokens:
			if self.is_url(tt) or self.is_email(tt) or not self.starts_with_letter(tt): # we don't want to split emojis, hashtags, etc..
				tokens.append(tt)
			else: # improve tokenization (for Italian or other languages different from English)
				tokens += self.it_tokenizer.tokenize(tt) # Italian tokenizer
		return tokens
		
	def build_distributional_docvec(self, list): # build a docvec using Doc2Vec (from gensim) algorithm over all the documents in the dataset
		documents = []
		for text_annotations in list:
			for annotation in text_annotations:
				id = annotation["text_annotation"]["id"]
				tokens = annotation["tokens"]
				documents.append( TaggedDocument( words=tokens, tags=[id] ) )
		return Doc2Vec(documents, vector_size=300, window=5, min_count=1, epochs=40, workers=flags.parallel_size) # 40 epochs
		
	def add_context_to_annotations(self, annotations):
		for awb in annotations:
			id = awb["text_annotation"]["id"]
			awb["distributional_docvec"] = self.doc2vec.docvecs[id]
			
	def add_wordvecs_to_annotations(self, annotations):
		for awb in annotations:
			awb["average_docvec"] = self.model.get_sentence_vector(awb["text"]) # average vector -> different from Gensim::Doc2Vec
			for t in awb["tokens_annotation"]:
				t["lemma_vector"] = self.model.get_word_vector(t["lemma"])
				t["token_vector"] = self.model.get_word_vector(t["token"])
			
	def get_pos_lemma(self, tag, token):
		if type(tag) is treetaggerwrapper.Tag:
			return tag.pos, tag.lemma
		return 'SYM', token # unknown symbol
			
	def get_OpeNER_pos(self, tag):
		# ABR ADJ ADV CON DET:def DET:indef FW INT LS NOM NPR NUM ORD PON PRE PRE:det PRO PRO:demo PRO:indef PRO:inter PRO:pers PRO:poss PRO:refl PRO:rela SENT SYM VER:cimp VER:cond VER:cpre VER:futu VER:geru VER:impe VER:impf VER:infi VER:pper VER:ppre VER:pres VER:refl:infi VER:remo
		if 'ADJ' in tag:
			return 'G'
		elif 'VER' in tag:
			return 'V'
		elif 'ADV' in tag:
			return 'A'
		elif 'NOM' in tag:
			return 'N'
		elif 'PRE' in tag:
			return 'P'
		else:
			return 'O'
			
	def get_WordNet_pos(self, tag): # <pos> is one of the module attributes ADJ, ADJ_SAT, ADV, NOUN or VERB
		# ABR ADJ ADV CON DET:def DET:indef FW INT LS NOM NPR NUM ORD PON PRE PRE:det PRO PRO:demo PRO:indef PRO:inter PRO:pers PRO:poss PRO:refl PRO:rela SENT SYM VER:cimp VER:cond VER:cpre VER:futu VER:geru VER:impe VER:impf VER:infi VER:pper VER:ppre VER:pres VER:refl:infi VER:remo
		if 'ADJ' in tag:
			return wn.ADJ
		elif 'VER' in tag:
			return wn.VERB
		elif 'ADV' in tag:
			return wn.ADV
		elif 'NOM' in tag:
			return wn.NOUN
		return wn.NOUN
		
	def is_interesting(self, tag):
		if 'ADJ' in tag:
			return True
		elif 'VER' in tag:
			return True
		elif 'ADV' in tag:
			return True
		elif 'NOM' in tag:
			return True
		elif 'PRE' in tag:
			return True
		return False
			
	def starts_with_letter(self, word):
		return word and word[0].isalpha()
		
	def is_url(self, url):
		return urlparse.urlparse(url).scheme != ""
		
	def is_email(self, mail):
		return validate_email(mail)
	
if not os.path.isfile(flags.preprocessed_dict+'.pkl'):
	app = SentiPolC()
	app.build_set()