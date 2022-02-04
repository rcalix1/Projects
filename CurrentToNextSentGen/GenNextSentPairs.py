######################################################
## 2021

import numpy as np
import nltk
import re

from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


######################################################

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

ps = PorterStemmer()


######################################################

f_in_covid  = open("data/covid/in/CovidCorpus.txt", "r")
f_in_pulm   = open("data/pulm/in/PulmCorpus.txt", "r")

f_out_sent1 = open("out/CovPulmCorpusSent1.txt", "w")
f_out_sent2 = open("out/CovPulmCorpusSent2.txt", "w")

######################################################

covid_list = f_in_covid.readlines()


for i, line in enumerate(covid_list):
    line = line.replace("\n", "")
    if i+1 >= len(covid_list):
        continue
    #print(covid_list[i])
    #print(covid_list[i+1])
    f_out_sent1.write(covid_list[i] )
    f_out_sent2.write(covid_list[i+1] )

######################################################

pulm_list = f_in_pulm.readlines()


for i, line in enumerate(pulm_list):
    line = line.replace("\n", "")
    if i+1 >= len(pulm_list):
        continue
 
    f_out_sent1.write(pulm_list[i] )
    f_out_sent2.write(pulm_list[i+1] )
    

#######################################################

f_in_covid.close()
f_in_pulm.close()

f_out_sent1.close()
f_out_sent2.close()

######################################################

print("<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>")
