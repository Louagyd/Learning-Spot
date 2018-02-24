# Word2Vec
making a **word embedding** and then test it using **skip-gram** model with python and tensorflow.

## Description of each file:

* [**Dataset**](Dataset/): the folder that contains the source data text. it is **_Penn Treebank_** dataset, known as **PTB** dataset which is widely used in machine learning of **NLP**.

* [**main.py**](main.py): the code which includes the implementation of the model using tensorflow, the training process, and saving model. 

* [**data.py**](data.py): defining some useful funtions. the main code has used some of these functions.

* [**AnalyseModel.py**](AnalyseModel.py): a code for showing some results and properties of the model that generated and saved by the main code.

## Details:

As you can see in the codes, the whole model is a **skip-gram** model. you can see more details about **skip-gram** in [here](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/).
in the main code, the **window-size** is set to **2**. for the higher values of **window-size**, the embedding efficiency is more, but also the training time would be very long.
the dimension of embedding space is set to **100**.
the **Loss** of the model is a **NCE Loss**. in tensorflow this loss can be used by **tf.nn.nce_loss(...)** function. this loss has some strengths like **Negative Sampling**.
for the training process, **batch size** is set to **128** and the **learning rate** is **0.001**. for optimization, an **Adam Optimizer** is trying to decrese the loss.
so after **50** epochs, we can have an embedding matrix. and thanks to **pickle**, the model is dumped and saved to a **.pkl** file.

## Results:

after training and saving the model, now its time to see some properties of this embedding.
the **most_similar_to_word** function that is defined in the **AnalyseModel.py**,  takes a word as an input word and return the nearest word to this input word in embedding space. you can see some examples:
>Most similar words:

>before ->  after

>prices ->  market

>adults ->  people

>month ->  week

>or ->  and

>dollar ->  market

>because ->  but

>more ->  less

>below ->  down

>us ->  them

another good intuition of this embedding is the scatter plot of words embedded in embedding space. first we should reduce the dimension of embedding space to 2 using **PCA**. then we can see the scatter plot of each word in this 2-dimensional space.
as you can see, similar words are close to eachother.

[scatter plot](Word2Vec.jpg)

