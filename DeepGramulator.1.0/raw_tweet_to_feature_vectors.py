## gramulator

import nltk
import re
import string
import operator
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords

#######################################################################################

stop_words_list = stopwords.words('english')

#################################################################

def return_list_of_terms(path):
    my_list = []
    file = open(path,'r')
    for line in file.readlines():
        word = line.replace("\n","")
        my_list.append(word)
    file.close()
    return my_list
    
##################################################################

def get_feature_auto(term_list,text):
    counter = 0
    for term in text:
        if term in term_list:
            counter = counter + 1
    return counter

##################################################################
##read frequency words

auto_yes_train = return_list_of_terms('/home/rcalix/Desktop/DeepGramulator/gramulator_features/train_auto_yes_gramulator_file.txt')
#auto_yes_test = return_list_of_terms('/home/rcalix/Desktop/DeepGramulator/gramulator_features/test_auto_yes_gramulator_file.txt')
auto_no_train = return_list_of_terms('/home/rcalix/Desktop/DeepGramulator/gramulator_features/train_auto_no_gramulator_file.txt')
#auto_no_test = return_list_of_terms('/home/rcalix/Desktop/DeepGramulator/gramulator_features/test_auto_no_gramulator_file.txt')

#######################################################################
## Deep grams

auto_yes_test = return_list_of_terms('/home/rcalix/Desktop/DeepGramulator/gramulator_features/testDeep_Grams/DeepGrams_test_auto_yes.txt')
auto_no_test = return_list_of_terms('/home/rcalix/Desktop/DeepGramulator/gramulator_features/testDeep_Grams/DeepGrams_test_auto_no.txt')

 
###################################################################

f_out = open('/home/rcalix/Desktop/DeepGramulator/output/test_set.features.3156_deep.txt','w')
file_in = '/home/rcalix/Desktop/DeepGramulator/data/3156_Test.csv'
   
###################################################

header = 'auto_yes_train,auto_no_train,auto_yes_test,auto_no_test,class\n'
f_out.write(header)

f_input = open(file_in,'r')
for line in f_input.readlines():
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
    the_class = temp[0]
    f1 = get_feature_auto(auto_yes_train,text)
    f2 = get_feature_auto(auto_no_train,text)
    f3 = get_feature_auto(auto_yes_test,text)
    f4 = get_feature_auto(auto_no_test,text)
    f_out.write("%s,%s,%s,%s,%s\n" % (f1,f2,f3,f4,the_class))
            
    
###################################################
f_input.close()
f_out.close()

###################################################

print '<<<<<<<<<<<<<DONE>>>>>>>>>>>'
