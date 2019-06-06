Generic Hierarchical Deep Reinforcement Learning for Sentiment Analysis
==========

The goal of this project is to experiment Hierarchical Reinforcement Learning applied to Sentiment Analysis. More details can be found in my Computer Science master thesis: [Deep Reinforcement Learning and sub-problem decomposition using Hierarchical Architectures in partially observable environments](https://amslaurea.unibo.it/16718/). 
For more details about the experiment, please read my master thesis.
  
This software is a fork of:
* [Miyosuda's UNREAL implementation](https://github.com/miyosuda/unreal)

This project has been tested on Debian 9 and macOS Mojave 10.14 with Python 3.7. The [setup.sh](setup.sh) script installs the necessary dependencies:
* [TreeTagger](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)
* [fastText](https://fasttext.cc/docs/en/crawl-vectors.html) pre-trained model for Italian
* [VU sentiment lexicon](https://github.com/opener-project/VU-sentiment-lexicon) by [OpeNER](http://www.opener-project.eu/)
* [emojipy](https://github.com/launchyard/emojipy) a Python library for working with emoji
* [NLTK](http://www.nltk.org/)
* [Googletrans](https://pypi.org/project/googletrans/2.2.0/), a free and unlimited python library that implemented Google Translate API.
* [gensim](https://radimrehurek.com/gensim/)
* [Tensorflow](https://www.tensorflow.org/)

Before running the [setup.sh](setup.sh) script you have to install: virtualenv, python3-dev, python3-pip and make. 
The [build.sh](build.sh) script pre-processes all the documents (of test and training set) in order to build the preprocessed.pkl file used (for efficiency purposes) during testing and training.
The [train.sh](train.sh) script starts the training.
The [test.sh](test.sh) script evaluates the trained agent using the weights in the most recent checkpoint. Please, remember that the trained policies are stochastic, thus the results obtained with [train.sh](train.sh) may slightly change each run.
In [A3C/options.py](A3C/options.py) you can edit the default algorithm settings.

During training the agent produces real-time statistics on the its perfomance. Among the statistics reported there are: 
* accuracy
* recall
* precision
* F1 score
* Matthews correlation coefficient

In the folder [checkpoint/backup](checkpoint/backup) there are the tensorflow checkpoints for the results described in the related paper. The default checkpoint gives the following F1 average scores: 0.72 for subjectivity and 0.70 for polarity.
In the folder [database](database) there are:
* Test and training set from [Evalita Sentipolc 2016](http://www.evalita.it/2016/tasks/sentipolc)
* The emoji sentiment data from [Emoji Sentiment Ranking](http://kt.ijs.si/data/Emoji_sentiment_ranking/)
* A pkl file built by [A3C/build.py](A3C/build.py) and containing the pre-processed documents of test and training set

For each thread, the statistics are printed as the average of the last 200 training episodes (documents used for training). The results.log file contains the average of the average of each thread.
Through the options.py file you can change most of the architecture parameters, including: the number of threads to use, whether to use the GPU or not, the initial learning rate, the log directories and much more.
The framework is composed of the following classes:
* Application ([server.py](A3C/agent/server.py)): the global A3C agent, which contains the methods for starting the local workers.
* Worker ([client.py](A3C/agent/client.py)): a local A3C worker.
* RMSPropApplier ([rmsprop_applier.py](A3C/model/rmsprop_applier.py)): the class for computing the gradient.
* ModelManager and A3CModel ([model_manager.py](A3C/model/model_manager.py) and [a3c_model.py](A3C/model/a3c_model.py)): within these classes the structure of the neural network is specified (LSTM, policy layer, value layer, CNN, FC, ecc..).
* Environment ([environment.py](A3C/environment/environment.py)): class that handles the interface between the agent and the environment. The Environment class has been extended with SentipolcEnvironment ([sentipolc_environment.py](A3C/environment/sentipolc_environment.py)). SentipolcEnvironment contains methods for calculating rewards, obtaining statuses and statistics on episodes, etc.

Citation
-------

Please use the following bibtex entry:
```
@mastersthesis{amslaurea16718,
	author    = "Francesco Sovrano",
	title     = "Deep Reinforcement Learning and sub-problem decomposition using Hierarchical Architectures in partially observable environments",
	school    = "Università di Bologna",
	year      = "2018",
	url = {http://amslaurea.unibo.it/16718/},
}
```

License
-------

This software is a fork of:
* [Miyosuda's UNREAL implementation](https://github.com/miyosuda/unreal)

Those parts of this software that are not inherited from the aforementioned repositories are released under the GPL v3.0 licence.