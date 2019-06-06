# -*- coding: utf-8 -*-
import tensorflow as tf
from agent.server import Application

def main(argv):
	app = Application()
	app.test()

if __name__ == '__main__':
	tf.app.run()