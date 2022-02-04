## NLG metrics

import re
import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize
from numpy import genfromtxt


import pickle
import collections


###################################################

from sklearn.metrics import jaccard_score
from rouge_score import rouge_scorer
from jiwer import wer

#import gensim.downloader as api
#model = api.load('word2vec-google-news-300')

#from bleurt import score
#bleurt_checkpoint = "bleurt/bleurt-base-128"
#bleurt_scorer = score.BleurtScorer(bleurt_checkpoint)


from bert_score import score as BERT_score


###################################################

path_in = "raw_data/pairs_pred_out.txt"
f_in = open(path_in, "r")

###################################################

string1 = f_in.read()



list_pairs_raw = []

string1 = string1.replace("#","")

print(string1)

list_pairs_raw = string1.split("BLEU score")

print(list_pairs_raw)

###################################################

f_in.close()

###################################################

def calc_BLEU_score(real, pred):
    real_ref  = nltk.word_tokenize(real)
    candidate = nltk.word_tokenize(pred)

    smoothie = nltk.translate.bleu_score.SmoothingFunction().method4

    BLEUscore = nltk.translate.bleu_score.sentence_bleu( [real_ref], candidate, smoothing_function=smoothie  )

    return BLEUscore * 100.0        ## Google multiples the score by 100

###################################################

def calc_ROGUE_score(real, pred):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(real, pred)
    return scores

###################################################

def calc_WER_score(real, pred):
    error = wer(real, pred)
    return error

###################################################

def meteor_score(real, pred):
    return nltk.translate.meteor_score.meteor_score(   [real], pred   )

###################################################

def distance_gensim_wmd(real, pred):
    distance = model.wmdistance(real, pred)
    return distance

###################################################

def calc_Bleurt(real, pred):
    references = [real]
    candidates = [pred]
    scores = bleurt_scorer.score(references=references, candidates=candidates)
    assert type(scores) == list and len(scores) == 1
    return scores


####################################################

def calc_BERT_score(real, pred):
    cands = [pred]
    refs  = [real]

    result = BERT_score(cands, refs, lang="en", verbose=True)
    return result
   

###################################################

pairs_dict = {}
index = 0
errors = 0 

try:
    for pair in list_pairs_raw:
        pair = pair.replace("\n","")
        temp = pair.split("real sent1")
        other_temp = temp[0].split("number")
        ref_id = other_temp[1]
        temp = temp[1].split("real sent2")
        real_sent1 = temp[0]
        temp = temp[1].split("pred sent2")
        real_sent2 = temp[0]
        pred_sent2 = temp[1]
        pairs_dict[index] = {}
        pairs_dict[index]['ref_id']     = str(ref_id)
        pairs_dict[index]['real_sent1'] = str(real_sent1)
        pairs_dict[index]['real_sent2'] = str(real_sent2)
        pairs_dict[index]['pred_sent2'] = str(pred_sent2)

        pairs_dict[index]['BLEU'] = 0 #str(calc_BLEU_score(real_sent2, pred_sent2))
        pairs_dict[index]['ROGUE'] = 0 #str(  calc_ROGUE_score(real_sent2, pred_sent2)  )
        pairs_dict[index]['BERTscore'] = str(  calc_BERT_score(real_sent2, pred_sent2)   )
        pairs_dict[index]['METEOR'] = 0 #str(meteor_score(real_sent2, pred_sent2))
        pairs_dict[index]['WER'] = 0 #str(  calc_WER_score(real_sent2, pred_sent2)   )
        pairs_dict[index]['Jaccard'] = 0
        pairs_dict[index]['Bleurt'] = 0 #str(  calc_Bleurt(real_sent2, pred_sent2)   )
        pairs_dict[index]['WMD'] = 0 #str(  distance_gensim_wmd(real_sent2, pred_sent2)    )
       
        index = index + 1
        print(index)
except:
    errors = errors + 1

print("stats")
print(index)
print(errors) 

###########################################################################

def write_dictionary(file_name, dict):
    with open(file_name, 'wb') as handle:
        pickle.dump(dict, handle)

###########################################################################

write_dictionary("output/metrics_only_BERTscore_dictionary.txt", pairs_dict )



##################################################

print("<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>")
