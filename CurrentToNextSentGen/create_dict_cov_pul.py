## create dict 
####################################################

import sklearn
import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import pickle
import random

####################################################

sent1_file = "data/SimilarText/Sent1.txt"
sent2_file = "data/SimilarText/Sent2.txt"

###################################################

sent1_sents = open(sent1_file).readlines()
sent2_sents = open(sent2_file).readlines()

###################################################

i = 0

train_count = 0
pairs_dict_train = {}

test_count   = 0
pairs_dict_test = {}


for line in sent1_sents:

    print("*********************************************")
    
    sent1_sent = sent1_sents[i]
    sent2_sent = sent2_sents[i]
    
    sent1_sent = sent1_sent.replace("\n", "")
    sent2_sent = sent2_sent.replace("\n", "")
    
    print(    sent1_sent   )
    print(    sent2_sent   )
    
    rand_n = random.randrange(100)                        # Integer from 0 to 99 inclusive
    if rand_n > 10.0:
        pairs_dict_train[train_count] = {}
        pairs_dict_train[train_count]['sent1'] = str(sent1_sent)
        pairs_dict_train[train_count]['sent2'] = str(sent2_sent)
        train_count = train_count + 1
    else:
        pairs_dict_test[test_count] = {}
        pairs_dict_test[test_count]['sent1'] = str(sent1_sent)
        pairs_dict_test[test_count]['sent2'] = str(sent2_sent)
        test_count = test_count + 1
        
    i = i + 1
  

 

    
####################################################

print("number of all pairs ", i )

print("train count ", train_count)

print("test count ",  test_count)


#####################################################


print(pairs_dict_test )

print(len(pairs_dict_train))
print(len(pairs_dict_test ))

######################################################
    
def load_dictionary(file_name):
    with open(file_name, 'rb') as handle:
        dict = pickle.loads(   handle.read()  )
    return dict

###########################################################################

def write_dictionary(file_name, dict):
    with open(file_name, 'wb') as handle:
        pickle.dump(dict, handle)

###########################################################################

write_dictionary("data/DictPairs/pairs_train_dictionary.txt", pairs_dict_train )
write_dictionary("data/DictPairs/pairs_test_dictionary.txt",  pairs_dict_test  )

###########################################################################


##train_      = load_dictionary("data/DictSimiCovPul/cov_pul_train_dictionary.txt")
##test_       = load_dictionary("data/DictSimiCovPul/cov_pul_test_dictionary.txt")

##print(train_)
##print(test_ )

print("train count ", train_count)

print("test count ",  test_count)


######################################################

print("<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>")
