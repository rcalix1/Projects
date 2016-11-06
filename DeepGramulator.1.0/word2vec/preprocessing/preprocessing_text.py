import re
import nltk
from nltk.corpus import stopwords

#####################################################################################

f_train = open("/home/rcalix/Desktop/DeepGramulator/word2vec/preprocessing/input/12559_Training.csv")
f_test  = open("/home/rcalix/Desktop/DeepGramulator/word2vec/preprocessing/input/3156_Test.csv")

string_train = f_train.read()
string_test = f_test.read()

#string = f.read()
string = string_train + string_test

string = re.sub(r'[^\x00-\x7F]', ' ', string) #remove unicodes
string = string.replace(","," ")
string = string.replace(".", " ")
string = string.replace("!", " ")
string = string.replace('"', " ")
string  =string.replace("\n"," ")
string = string.replace("\t"," ") 
string = re.sub('\s+', ' ', string)
string = re.sub('[^0-9a-zA-Z]+', ' ', string)
print string 
#x = raw_input()

list_of_words2 = []
list_of_words = string.split(" ")
for upper_word in list_of_words:
    if upper_word not in stopwords.words('english'):
        list_of_words2.append(upper_word.lower())

string1 = " ".join(list_of_words2)

vocabulary = []
for word in list_of_words2:
    #print word
    if word not in vocabulary:
        vocabulary.append(word)

print vocabulary

#x = raw_input()
print string1

f_out = open("/home/rcalix/Desktop/DeepGramulator/word2vec/preprocessing/output/word2vec_input_12559_3156_combined.csv","w")
f_out.write(string1)
f_out.close()
print "size is: ", len(vocabulary)


print "<<<<<<<<<<<DONE>>>>>>>>>>"
## 26940  
