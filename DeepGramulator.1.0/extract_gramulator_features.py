## gramulator

import nltk
import re
import string
import operator
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords

#######################################################################################

file_input='/home/rcalix/Desktop/DeepGramulator/data/3156_Test.csv'

stop_words_list = stopwords.words('english')

########################################################################################


dictionary_freq_yes = {}
dictionary_freq_no = {}

f_frequency_count = open(file_input,'r')

#######################################################################################

for line in f_frequency_count.readlines():
    print line
    #rr = raw_input()
    temp = line.split(",")
    parts = len(temp)
    num_parts = int(parts) 
    tweet_part = temp[4:num_parts]
    tweet_string = ' '.join(tweet_part)
    tweet_string = re.sub(r'[^\x00-\x7f]',r' ',tweet_string)
    text_temp = word_tokenize(tweet_string)
    text = [i for i in text_temp if i not in stop_words_list]
    if temp[0] in ["Yes","Unsure"]:
        for word in text: 
            word = word.lower()
            if word in dictionary_freq_yes:
                dictionary_freq_yes[word] = dictionary_freq_yes[word] + 1
            else:
                dictionary_freq_yes[word] = 1

    if temp[0] in ["No"]:
        for word in text: 
            word = word.lower()
            if word in dictionary_freq_no:
                dictionary_freq_no[word] = dictionary_freq_no[word] + 1
            else:
                dictionary_freq_no[word] = 1

##################################################################

f_frequency_count.close()


##################################################################

words_dictionary_freq_no_sorted = []
words_dictionary_freq_yes_sorted = []

###################################################################

dictionary_freq_no_sorted = sorted(dictionary_freq_no.items(), key=operator.itemgetter(1) , reverse=True)
for tup in dictionary_freq_no_sorted:
    words_dictionary_freq_no_sorted.append(tup[0])


dictionary_freq_yes_sorted = sorted(dictionary_freq_yes.items(), key=operator.itemgetter(1) , reverse=True)
for tup in dictionary_freq_yes_sorted:
    words_dictionary_freq_yes_sorted.append(tup[0])

############################################################################

list_automatic_freq_yes = []
list_automatic_freq_no = []

############################################################################

for word in words_dictionary_freq_no_sorted[0:11000]:
    if word not in words_dictionary_freq_yes_sorted:
        list_automatic_freq_no.append(word.lower())

for word in words_dictionary_freq_yes_sorted[0:11000]:
    if word not in words_dictionary_freq_no_sorted:
        list_automatic_freq_yes.append(word.lower())

##############################################################################

print list_automatic_freq_yes
rr = raw_input()
print list_automatic_freq_no
rr = raw_input()


##########################################################################################

f_zz_auto_yes_list = open('/home/rcalix/Desktop/DeepGramulator/gramulator_features/test_auto_yes_gramulator_file.txt','w')
f_zz_auto_no_list = open('/home/rcalix/Desktop/DeepGramulator/gramulator_features/test_auto_no_gramulator_file.txt','w')


for item in list_automatic_freq_no:
    f_zz_auto_no_list.write(item)
    f_zz_auto_no_list.write('\n')

for item in list_automatic_freq_yes:
    f_zz_auto_yes_list.write(item)
    f_zz_auto_yes_list.write('\n')
        
f_zz_auto_yes_list.close() 
f_zz_auto_no_list.close()
   
            
    
##################################################3

print '<<<<<<<<<<<<<DONE>>>>>>>>>>>'
