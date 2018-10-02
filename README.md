Deep Reinforcement Learning for Sentiment Analysis with (partitioned) A3C
==========
  
This software is a fork of:
* [Miyosuda's UNREAL implementation](https://github.com/miyosuda/unreal)

This project has been tested on Debian 9. The setup.sh script installs the necessary dependencies:
* [TreeTagger](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)
* [fastText](https://fasttext.cc/docs/en/crawl-vectors.html) pre-trained model for Italian
* [VU sentiment lexicon](https://github.com/opener-project/VU-sentiment-lexicon) by [OpeNER](http://www.opener-project.eu/)
* [emojipy](https://github.com/launchyard/emojipy) a Python library for working with emoji
* [NLTK](http://www.nltk.org/)
* [Googletrans](https://pypi.org/project/googletrans/2.2.0/), a free and unlimited python library that implemented Google Translate API.
* [gensim](https://radimrehurek.com/gensim/)
* [Tensorflow](https://www.tensorflow.org/)

Before running setup.sh you must have installed virtualenv, python3-dev, python3-pip and make. 
For more details, please read the related paper.

The build.sh script pre-processes all the documents (of test and training set) in order to build the preprocessed.pkl file used (for efficiency purposes) during testing and training.
The train.sh script starts the training.
The test.sh script evaluates the trained agent using the weights in the most recent checkpoint. Please, remember that the trained policies are stochastic, thus the results obtained with train.sh may slightly change at each run.

During training the agent produces real-time statistics on the its perfomance. Among the statistics reported there are: 
* accuracy
* recall
* precision
* F1 score
* Matthews correlation coefficient

In the folder "checkpoint/backup" there are the tensorflow checkpoints for the results described in the related paper. The default checkpoint gives the following F1 average scores: 0.72 for subjectivity and 0.70 for polarity.
In the folder "database" there are:
* Test and training set from [Evalita Sentipolc 2016](http://www.evalita.it/2016/tasks/sentipolc)
* The emoji sentiment data from [Emoji Sentiment Ranking](http://kt.ijs.si/data/Emoji_sentiment_ranking/)
* A pkl file build by "build.py" and containing the pre-processed documents of test and training set

For each thread, the statistics are printed as the average of the last 200 training episodes (documents used for training). The results.log file contains the average of the average of each thread.
Through the options.py file you can change most of the architecture parameters, including: the number of threads to use, whether to use the GPU or not, the initial learning rate, the log directories and much more.
The framework is composed of the following classes:
* Application (train.py): the global A3C agent, which contains the methods for starting the local workers.
* Trainer (trainer.py): a local A3C worker.
* RMSPropApplier (rmsprop_applier.py): the class for asynchronously computing the gradient.
* MultiAgentModel and A3CModel (multi_agent_model.py and a3c_model.py): within these classes the structure of the neural network is specified (LSTM, policy layer, value layer, CNN, FC, ecc..).
* Environment (environment.py): class that handles the interface between the agent and the environment. The Environment class has been extended with SentipolcEnvironment (sentipolc_environment.py). SentipolcEnvironment contains methods for calculating rewards, obtaining statuses and statistics on episodes, etc.

License
-------

This software is a fork of:
* [Miyosuda's UNREAL implementation](https://github.com/miyosuda/unreal)

Those parts of this software that are not inherited from the aforementioned repositories are released under the GPL v3.0 licence.