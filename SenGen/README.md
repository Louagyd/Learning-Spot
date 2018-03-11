# SenGen
predicting the next word and generating sentences using dynamic lstm rnn with python and tensorflow.

## Description of each file:

* [**Dataset**](Dataset/): the folder that contains the train sources. for this project we have used 2 sources. one is the famous *PTB Dataset* and the other one is *MultiNLI* dataset.

* [**GensimModels**](GensimModels/): the folder where the generated gensim models will be stored.

* [**Models**](Models/): the folder where the generated RNN models will be saved.

* [**gensimmodel.py**](gensimmodel.py): for generating gensim model using the source sentences data.

* [**rnn.py**](rnn.py): the main code that contains the training process implemented with tensorflow library. then saving the model in the associated folder.

* [**Conversation.py**](Conversation.py): the code for generating sentence using the saved RNN model.

## Details:

first thing that I want to talk about is the data sources. as you can see from above, we have used two sources for the training process. but the results that you can see later, are associated with **MultiNLI** dataset.
you can see more details and download it from [here](https://www.nyu.edu/projects/bowman/multinli/).
after extracting this data, now we have a set of sentences that we should make a **gensim** model with them.
also you can read about gensim models and see the tutorial from [here](https://radimrehurek.com/gensim/models/word2vec.html).
eventually we embed each word in a 100 dimensianal space using gensim. now the next step is to make a dynamic recurrent Neural Network for predicting the next word.
for controling the maximum length of input sequences, first we filter sentences and we just keep sentences with less than or equal to 20 words.
now we build a dynamic rnn with a two layer lstm cell with 128 nodes, and dropped with 0.8 as keep probobality.
then we should save the generated model then we can generate sentence with this model in [Conversation.py](Conversation.py).

## Results:
some generated sentences using this model. first bold words are initial words and the rest of sentence are generated words.

* **I wish I have a good** job but yourself will allow a problem that could meet.
* **please do not forget to bring** it and just just a cat I want money actually.
* **a good business man is always** really problem it yeah they will say that just uh.
* **where should I go after** uh kids could change the it actually know it wasn't.
* **there are many requests waiting for** programs um activities in experties because company consider really yeah.
