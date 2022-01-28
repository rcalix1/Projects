import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from gensim.test.utils import datapath
from gensim import utils


##############################
##  check out the text input format

'''
https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/test/test_data/lee_background.cor
'''

## /home/maquina1/Desktop/BatesFungi/output

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('/home/maquina1/Desktop/BatesFungi/output/all_text_combined.txt')
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)

##############################

sentences = MyCorpus()
model = Word2Vec(sentences=sentences, vector_size=128)

##############################

#fungi_sentences = LineSentence(fungi_text)
#model = Word2Vec(sentences=fungi_text, vector_size=128, window=4, min_count=10, workers=4)

###############################

print(  model.wv['fungus']   )

##############################
'''
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


x_vals, y_vals, labels = reduce_dimensions(model)



def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    print("here")
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

    plt.show()



plot_with_matplotlib(x_vals, y_vals, labels)
'''
##########################################

for index, word in enumerate(model.wv.index_to_key):
    if index == 4000:
        break
    print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")

##########################################

f = open("genus_final_list.txt", 'r')

sim_i = 0
sim_errors = 0

for line in f.readlines():
    #print(line)
    temp = line.split(":")
    #print(temp[0])
    word_genus = temp[0].lower()
    #print(  word_genus   )

    try:
        
        print(model.wv.most_similar(positive=[word_genus], topn=20))
        print(word_genus)
        print("**************************************")
        sim_i = sim_i + 1 
    except:
        sim_errors = sim_errors + 1
    
    

print("sim errors", sim_errors   )
print("total genus found in corpus", sim_i)

##########################################

print('<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>')
