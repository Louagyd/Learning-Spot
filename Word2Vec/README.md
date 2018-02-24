# Word2Vec
making a **word embedding** and then test it using **skip-gram** model with python and tensorflow.

## Description of each file:

* [**Dataset**](Dataset/): the folder that contains the source data text. it is **_Penn Treebank_** dataset, known as **PTB** dataset which is widely used in machine learning of **NLP**.

* [**main.py**](main.py): the code which includes the implementation of the model using tensorflow, the training process, and saving model. 

* [**data.py**](data.py): defining some useful funtions. the main code has used some of these functions.

* [**AnalyseModel.py**](AnalyseModel.py): a code for showing some results and properties of the model that generated and saved by the main code.

## Details:

As you can see in the codes, the whole model is a **skip-gram** model. you can see more details about **skip-gram** in [here](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/).
in the main code, the **window-size** is set to **1**. so every input and target of the neural network are one-hots of two adjacent words. for the higher values of **window-size**, the embedding efficiency is more, but also the training time would be very long.
