import gensim
import jsonlines as js

# generate a gensim model using the sentences data
# this function seperates all sentences and add a <eos> word to the last of each sentence. then generates a gensim model
def generate_model(data, name, min_count, size, json = False):
    if json:
        sen_data = js.open(data)
        sentences = []
        lens = []
        for obj in sen_data:
            this_sentence = obj['sentence1'].split()
            this_len = len(this_sentence)
            this_sen = []
            this_sen[0:this_len] = this_sentence
            this_sen.append('<eos>')
            sentences.append(this_sen)
            lens.append(len(this_sen))
    else:
        with open(data) as f:
            sen_data = f.readlines()
            sentences = []
            lens = []
            for line in sen_data:
                this_sentence = line.split()
                this_len = len(this_sentence)
                this_sen = []
                this_sen[0:this_len] = this_sentence
                this_sen.append('<eos>')
                sentences.append(this_sen)
                lens.append(len(this_sen))


    model = gensim.models.Word2Vec(sentences, min_count=min_count, size=size, workers=4)
    model.save(('GensimModels/'+name))
    return sentences, lens