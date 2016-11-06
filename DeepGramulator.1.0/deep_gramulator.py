import nltk
import re
import string
import operator
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords



file1='/home/osboxes/Desktop/twitter_nih/temp_input.txt'
output_file = '/home/osboxes/Desktop/twitter_nih/pnw_nih_results_128d.csv'


##########################################################################################
## 0 no, dont read word lists from file. Creates word list files
## 1 yes, read words lists from file. Does not create word list files

read_word_lists_from_file = 0

#########################################################################################

stemmer = PorterStemmer()

########################################################################################

from_file_list_automatic_freq_no = []
from_file_list_automatic_freq_yes = []

#########################################################################################
##'/home/osboxes/Desktop/twitter_nih/word_lists/'
if read_word_lists_from_file == 1:
    f_zz_auto_yes_list = open('/home/osboxes/Desktop/twitter_nih/word_lists/auto_yes_list_file.txt','r')
    f_zz_auto_no_list = open('/home/osboxes/Desktop/twitter_nih/word_lists/auto_no_list_file.txt','r')


    

    for word in f_zz_auto_yes_list.readlines():
        word = word.replace('\n','')
        from_file_list_automatic_freq_yes.append(word)

    for word in f_zz_auto_no_list.readlines():
        word = word.replace('\n','')
        from_file_list_automatic_freq_no.append(word)

        
    f_zz_auto_yes_list.close() 
    f_zz_auto_no_list.close()
    f_zz_twitter_client_list.close()
    


##########################################################################################

stop_words_list = stopwords.words('english')


all_words_string = ''
f_frequency = open(file1,'r')
dictionary_freq = {}
for line in f_frequency.readlines():
    temp = line.split(",")
    parts = len(temp)
    num_parts = int(parts) 
    temp1 = temp[10:num_parts]
    tweet_string = ' '.join(temp1)
    tweet_string = re.sub(r'[^\x00-\x7f]',r' ',tweet_string)
    text = word_tokenize(tweet_string)
    all_words_string = all_words_string + ' ' + tweet_string
    #print temp
    if temp[4] not in all_twitter_clients:
        all_twitter_clients.append(temp[4])
    if temp[9] not in all_user_ids:
        all_user_ids.append(temp[9])

all_tokens = word_tokenize(all_words_string)
unique_words = set(all_tokens)
#print unique_words
#print all_words_string
print len(all_words_string)
print len(unique_words)
#x = raw_input()
f_frequency.close()

######################################################################################


dictionary_freq_yes = {}
dictionary_freq_no = {}

f_frequency_count = open(file1,'r')

for line in f_frequency_count.readlines():
    temp = line.split(",")
    parts = len(temp)
    num_parts = int(parts) 
    temp1 = temp[10:num_parts]
    tweet_string = ' '.join(temp1)
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




words_dictionary_freq_no_sorted = []
words_dictionary_freq_yes_sorted = []


print "dictionary_freq_no"
dictionary_freq_no_sorted = sorted(dictionary_freq_no.items(), key=operator.itemgetter(1) , reverse=True)
for tup in dictionary_freq_no_sorted:
    words_dictionary_freq_no_sorted.append(tup[0])

#print dictionary_freq_no_sorted[1200:1600]


print "   "
print "dictionary_freq_yes"
dictionary_freq_yes_sorted = sorted(dictionary_freq_yes.items(), key=operator.itemgetter(1) , reverse=True)
#print dictionary_freq_yes_sorted #[0:400]

for tup in dictionary_freq_yes_sorted:
    words_dictionary_freq_yes_sorted.append(tup[0])



list_automatic_freq_yes = []
list_automatic_freq_no = []

for word in words_dictionary_freq_no_sorted[0:11000]:
    if word not in words_dictionary_freq_yes_sorted:
        list_automatic_freq_no.append(stemmer.stem(word.lower()))

for word in words_dictionary_freq_yes_sorted[0:11000]:
    if word not in words_dictionary_freq_no_sorted:
        list_automatic_freq_yes.append(stemmer.stem(word.lower()))

print list_automatic_freq_yes
#x = raw_input()
print list_automatic_freq_no

#x = raw_input()

f_frequency_count.close()
##########################################################################################

def func_tags_NNP(tag_sequence):
    count_nnp = 0
    for tup in tag_sequence:
        if tup[1] == 'NNP':
            count_nnp = count_nnp + 1
    return count_nnp

##########################################################################################
def func_tags_PRP(tag_sequence):
    count_prp = 0
    for tup in tag_sequence:
        if tup[1] == 'PRP':
            count_prp = count_prp + 1
    return count_prp

##########################################################################################

if read_word_lists_from_file == 1:
    all_twitter_clients = from_file_all_twitter_clients[:] 
    list_automatic_freq_no = from_file_list_automatic_freq_no[:]
    list_automatic_freq_yes = from_file_list_automatic_freq_yes[:]
    print all_twitter_clients 
    print list_automatic_freq_no
    #x = raw_input()
    print list_automatic_freq_yes
    #x = raw_input()


##########################################################################################

def automatic_freq_no(text):
    count_words = 0
    
    for word in text:
        word1 = stemmer.stem(word.lower())
        if word1 in list_automatic_freq_no:
            count_words = count_words + 1
    return count_words
    
########################################################################################

def automatic_freq_yes(text):
    count_words = 0
    
    for word in text:
        word1 = stemmer.stem(word.lower())
        if word1 in list_automatic_freq_yes:
            count_words = count_words + 1
    return count_words


##########################################################################################

def func_No_freq_terms(text):
    count_words = 0
    no_freq_terms = ['depression', 'offers','natural','looking', 'remedies','choose','price','get','special','compare',
                     'body','recommended','save','effective','offer','sale', 'deal','ebay','stock','nutrition','selling',
                     'quality','wellness','miss','savings','active','prices','relieve','deals','calming','$','youtube',
                     'buy','pay','bionutrients','naturally', 'disorders','reddit']
    for word in text:
        if word in no_freq_terms:
            count_words = count_words + 1
    return count_words
        
##########################################################################################

def func_Yes_freq_terms(text):
    count_words = 0
    yes_freq_terms = ['helpful', 'wonders','good','cat','enough','said','jesus','mood','started','thoughts','quite','they','feel',
                      'aware','idea','recommended','well','crazy', 'effect', 'helps', 'ever','react', 'effects','side','quit',
                      'might', 'placebo', 'mild', 'right','stuff', 'trying', 'meds', 'felt','happy', 'hopefully', 'worse',
                      'much', 'depressed', 'possibly', 'ago','past', 'sleep', 'pills', 'something', 'thinking','this', 'tried',
                      'helped', 'found', 'issues','talk', 'girl', 'helping', 'work', 'my', 'actually', 'seems', 'boy', 'better',
                      'relief', 'badly', 'since', 'going','took','coffee', 'lol', 'relaxation', 'little', 'hello', 'anxiety',
                      'caffeine', 'sunlight', 'root', 'thank', 'shit', 'feeling', 'worked', 'think', 'though', 'life', 'sun',
                      'stress', 'bit', 'calming' ]
    for word in text:
        if word in yes_freq_terms:
            count_words = count_words + 1
    return count_words

##########################################################################################

def create_files_of_word_lists():
    f_zz_auto_yes_list = open('/home/osboxes/Desktop/twitter_nih/word_lists/auto_yes_list_file.txt','w')
    f_zz_auto_no_list = open('/home/osboxes/Desktop/twitter_nih/word_lists/auto_no_list_file.txt','w')
    f_zz_twitter_client_list = open('/home/osboxes/Desktop/twitter_nih/word_lists/twitter_client_list_file.txt','w')

    for item in all_twitter_clients:
        f_zz_twitter_client_list.write(item)
        f_zz_twitter_client_list.write('\n')

    for item in list_automatic_freq_no:
        f_zz_auto_no_list.write(item)
        f_zz_auto_no_list.write('\n')

    for item in list_automatic_freq_yes:
        f_zz_auto_yes_list.write(item)
        f_zz_auto_yes_list.write('\n')
        
    f_zz_auto_yes_list.close() 
    f_zz_auto_no_list.close()
    f_zz_twitter_client_list.close()
    
            
    
##########################################################################################


header_str = 'id,followers_count,friends_count,words_tweet,class'

f_output.write(header_str)
f_output.write("\n")
print header_str

###################################################################################

for line in f.readlines():
    print line
    the_vector = {}
    #tweet_string = re.sub(r'[^\x00-\x7f]',r' ',tweet_string)
    text = word_tokenize(tweet_string)
    print text
    #rr = raw_input()

    the_vector['class'] = ?
    the_vector['id_val'] = ?  

    the_vector['auto_No_freq_terms'] = automatic_freq_no(text)
    the_vector['auto_Yes_freq_terms'] = automatic_freq_yes(text)
 
  

    f_output.write(',')
    f_output.write(str(the_vector['auto_Yes_freq_terms']))
    f_output.write(',')
    f_output.write(str(the_vector['auto_No_freq_terms']))
    f_output.write(',')
    f_output.write(str(the_vector['twitter_client']))
    f_output.write(',')
    f_output.write(str(the_vector['user_id']))
    f_output.write(',')
    
    if the_vector['class'] in ["Yes", "No", "Unsure"]:
        f_output.write(the_vector['class'])
    else:
        f_output.write("Unsure")
    f_output.write("\n")
    #x = raw_input()

############################################################################3

print '<<<<<<<<<<<<<DONE>>>>>>>>>>>'
