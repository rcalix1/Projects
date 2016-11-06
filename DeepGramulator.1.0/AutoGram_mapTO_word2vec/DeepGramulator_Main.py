
import numpy as np
#from sklearn.cross_validation import train_test_split
#from sklearn.preprocessing import StandardScaler
import nltk
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import pandas as pd
#from sklearn.preprocessing import LabelEncoder
#from sklearn import decomposition

from nltk.stem.porter import *

#############################################################################
## set parameters

np.set_printoptions(threshold=np.inf) ## print all values in numpy array

#############################################################################

stemmer = PorterStemmer()

#############################################################################

def get_list(path):
    the_list = []
    f = open(path,'r')
    for line in f.readlines():
        word = line.replace("\n","")
        if word not in the_list:
            the_list.append(word)
    return the_list


#############################################################################
## load problem data
path = '/home/rcalix/Desktop/DeepGramulator/AutoGram_mapTO_word2vec/input/word2vec_space/vector_space.txt'
df = pd.read_csv(path, sep=',')

Vectors = df.ix[:,1:128]
Words = df.ix[:,0]

numpyMatrix = Vectors.as_matrix()


##########################################################################
## similarity


similarity = np.asarray(np.asmatrix(numpyMatrix) * np.asmatrix(numpyMatrix).T )
similarity_df = pd.DataFrame(similarity)
print pd.DataFrame(similarity, index=Words, columns=Words).ix[1:10,1:10]

########################################################################

auto_yes_train = get_list('/home/rcalix/Desktop/DeepGramulator/AutoGram_mapTO_word2vec/input/auto_grams/train_auto_yes_gramulator_file.txt')
auto_no_train = get_list('/home/rcalix/Desktop/DeepGramulator/AutoGram_mapTO_word2vec/input/auto_grams/train_auto_no_gramulator_file.txt')

auto_yes_train_stemmed = [stemmer.stem(term) for term in auto_yes_train]
auto_no_train_stemmed = [stemmer.stem(term) for term in auto_no_train]


#########################################################################

print auto_yes_train

indices_auto_yes_train = [int(i) for i in range(len(auto_yes_train)) if stemmer.stem(Words[i].lower()) in auto_yes_train]
indices_auto_no_train = [int(i) for i in range(len(auto_no_train)) if stemmer.stem(Words[i].lower()) in auto_no_train]

##########################################################################

#test_indices_list = [3, 5, 8]

selected_rows_auto_yes_train = similarity_df.ix[indices_auto_yes_train] #test_indices_list]
selected_rows_auto_no_train = similarity_df.ix[indices_auto_no_train]

#print selected_rows_auto_yes_train

##########################################################################

#top_n = 5
#result_auto_train_yes = pd.DataFrame(
 #    {n: selected_rows_auto_yes_train.T[col].nlargest(top_n).index.tolist() for n, col in enumerate(selected_rows_auto_yes_train.T)}).T


##########################################################################


def get_results_list(indices_auto_train):
    top_n = 10
    result_test_auto_similar = []

    for index in indices_auto_train:
        print index
        row = similarity_df.ix[index]
        row_list = row.values.tolist()
        vals = np.array(row_list)
        sort_index = np.argsort(vals)
        start_index = len(Words) - top_n
        end_index = len(Words)
        top_indeces = sort_index[start_index:end_index]
        Selected_words = Words[top_indeces]
        for word in Selected_words:
            if word not in auto_yes_train:
                if word not in auto_no_train:
                    if word not in nltk.corpus.stopwords.words('english'): 
                        if word not in result_test_auto_similar:
                            result_test_auto_similar.append(word)

    return result_test_auto_similar


####################################################################################

def create_output_text_files(path, the_list):
    f_out = open(path, 'w')
    for word in the_list:
        f_out.write(word+'\n')
    f_out.close()

####################################################################################

result_test_auto_yes_similar = get_results_list(indices_auto_yes_train)
result_test_auto_no_similar = get_results_list(indices_auto_no_train)
     
####################################################################################

path_out = '/home/rcalix/Desktop/DeepGramulator/AutoGram_mapTO_word2vec/output/'
create_output_text_files(path_out + 'DeepGrams_test_auto_yes.txt', result_test_auto_yes_similar)
create_output_text_files(path_out + 'DeepGrams_test_auto_no.txt', result_test_auto_no_similar[0:2000])  ##reduced nos

####################################################################################

print len(indices_auto_yes_train)
print len(indices_auto_no_train)
print len(Words)


##########################################################################

print "<<<<<<<<<<DONE>>>>>>>>>>>>"
