import nltk
import re
import string
import operator
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
import collections
import math
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
import zipfile
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
import gensim, logging
import urllib2 as urllib
import sys
import simplejson as json
import csv
import os
import time
import re
from datetime import datetime
import jsonpickle, operator,json
import oauth2 as oauth
import urllib2 as urllib

import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE


#############################################################################
## set parameters

np.set_printoptions(threshold=np.inf) ## print all values in numpy array
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#############################################################################

stemmer = PorterStemmer()
stop_words_list = stopwords.words('english')


##############################################################################

def run_gensim(sentences):
    model = gensim.models.Word2Vec(sentences, min_count=1,size=128)
		

################################################################################################

def build_dataset(words):
  count = [['UNK', -1]]
  print count

  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)

  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)

  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 

  return data, count, dictionary, reverse_dictionary



#####################################################################################################

##function to generate a training batch for the skip-gram model


def generate_batch( batch_size, num_skips, skip_window):
  global data_index
  #global data
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  #print batch
  #x =raw_input()
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  #print labels
  #x = raw_input()
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  #print buffer
  #x = raw_input()
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
    #print buffer
    #print "index: ", data_index
    #x = raw_input()
  for i in range(batch_size // num_skips):  ## range( 4 or 2) 
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
   
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels



################################################
## read data into a string


def read_data2(filename):
    f = open(filename)
    data = tf.compat.as_str(f.read()).split()
    return data


##########################################################################################

def write_vector_space_to_file(embeddings, labels):
  f_vector_space = open("output/vector_space.txt","w")
  features = []
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  for i, label in enumerate(labels):
    features = embeddings[i,:]
    features_string = ""
    for feat in features:
        features_string = features_string + str(feat) + ","
    string = str(label) + "," + features_string
    f_vector_space.write(string)
    f_vector_space.write("\n")
  f_vector_space.close()   
   



############################################################################


def automatic_freq(text, list_of_words):
    count_words = 0
    for word in text:
        if word in list_of_words:
            count_words = count_words + 1
    return count_words


########################################################################################


def combine_train_test_for_word2vec(train_text_list, test_text_list):
    super_list = train_text_list + test_text_list
    path = 'output/combinedText.csv'
    f = open(path,'w')
    for sentence in super_list:
        line = " ".join(sentence)
        line = re.sub("\n","",line)
        f.write(line + " ")
    f.close()
    return super_list


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


def create_frequency_counts(X_strings, y_class, n=11000):
    dictionary_freq_yes = {}
    dictionary_freq_no = {}
    for i in range(len(X_strings) - 1):
        all_tokens = X_strings[i]
        the_class = y_class[i]
        #unique_words = set(all_tokens)
        if the_class in ["Yes","Unsure"]:
            for word in all_tokens:
                if word in dictionary_freq_yes:
                    dictionary_freq_yes[word] = dictionary_freq_yes[word] + 1
                else:
                    dictionary_freq_yes[word] = 1

        if the_class in ["No"]:
            for word in all_tokens:
                if word in dictionary_freq_no:
                    dictionary_freq_no[word] = dictionary_freq_no[word] + 1
                else:
                    dictionary_freq_no[word] = 1

    words_dictionary_freq_no_sorted = []
    words_dictionary_freq_yes_sorted = []

    dictionary_freq_no_sorted = sorted(dictionary_freq_no.items(), key=operator.itemgetter(1) , reverse=True)
    for tup in dictionary_freq_no_sorted:
        words_dictionary_freq_no_sorted.append(tup[0])

    dictionary_freq_yes_sorted = sorted(dictionary_freq_yes.items(), key=operator.itemgetter(1) , reverse=True)
    for tup in dictionary_freq_yes_sorted:
        words_dictionary_freq_yes_sorted.append(tup[0])

    list_automatic_freq_yes = []
    list_automatic_freq_no = []

    for word in words_dictionary_freq_no_sorted[0:n]:
        if word not in words_dictionary_freq_yes_sorted:
            list_automatic_freq_no.append(word)

    for word in words_dictionary_freq_yes_sorted[0:n]:
        if word not in words_dictionary_freq_no_sorted:
            list_automatic_freq_yes.append(word)

    return list_automatic_freq_yes, list_automatic_freq_no



##########################################################################################


def get_text_from_file(path):
    list_of_classes = []
    list_of_tweets = []
    f_open = open(path,'r')
    for line in f_open.readlines():
        temp = line.split(",")
        num_parts = int(len(temp))
        the_class = temp[0]
        text_temp = temp[4:num_parts]
        tweet_string = ' '.join(text_temp)
        tweet_string = re.sub(r'[^\x00-\x7f]',r' ',tweet_string)
        #string = [i.lower() for i in tweet_string if i not in stop_words_list]
        string = re.sub(r'[^\x00-\x7F]', ' ', tweet_string) #remove unicodes
        string = string.replace(","," ")
        string = string.replace(".", " ")
        string = string.replace("!", " ")
        string = string.replace('*', " ")
        string = string.replace('"', " ")
        string  =string.replace("\n"," ")
        string = string.replace("\t"," ")
        string = re.sub('\s+', ' ', string)
        string = re.sub('[^0-9a-zA-Z]+', ' ', string)
        #tweet_string = re.sub(r'[^\x00-\x7f]',r' ',tweet_string)
        tokens = word_tokenize(string)
        string_as_tokens = [stemmer.stem(i.lower()) for i in tokens if i not in stop_words_list]
        list_of_classes.append(the_class)
        list_of_tweets.append(string_as_tokens)
    f_open.close()
    return list_of_tweets, list_of_classes


##########################################################################################


def get_word_embedding_128(dictionary_vectors, text):
    embedding = np.zeros(128)
    count = 0.000001
    for word in text:
        if word in dictionary_vectors:
            vector = np.array(dictionary_vectors[word])
            embedding = embedding + vector
            count = count + 1
    avg_embedding = embedding / count
    x_arrstr = np.char.mod('%f', avg_embedding)
    x_str = ",".join(x_arrstr)
    return x_str, avg_embedding
        


#########################################################################################

def dot_product2(v1,v2):
    return sum(map(operator.mul, v1,v2))


##########################################################################################

def vector_cos5(v1,v2):
    prod = dot_product2(v1,v2)
    len1 = math.sqrt(dot_product2(v1,v1))
    len2 = math.sqrt(dot_product2(v2,v2))
    return prod / (len1 * len2)

##########################################################################################

def get_average_of_list(list_words, dictionary_vectors):
    embedding = np.zeros(128)
    count = 0.0000001
    for word in list_words:
        if word in dictionary_vectors:
            vector = np.array(dictionary_vectors[word])
            embedding = embedding + vector
            count = count + 1
    avg_embedding = embedding / count
    return avg_embedding

#########################################################################################

def sim_tweet_to_avg_list(numpy_word_embedding, avg_embedding):
    #sim = vector_cos5(embedding,numpy_word_embedding)
    sim = np.linalg.norm(avg_embedding-numpy_word_embedding)
    return sim

##########################################################################################

def sim_avg_no_avg_yes(embedding1,embedding2):
    sim = np.linalg.norm(embedding1-embedding2)
    return sim

##########################################################################################


def get_heading():
    s1= "ygram,ngram,ydeep,ndeep,ysimgram,nsimgram,ysimdeep,nsimdeep,simavggram,simavgdeep,"
    s2 = ''
    for i in range(128):
        s2 = s2 + "e" + str(i) + ","
    s3 = s1 + s2 + "class" + "\n"
    return s3



##########################################################################################



def create_feature_vectors(path, text_list, list_of_classes, auto_yes_train,auto_no_train,auto_yes_test, auto_no_test):
    dictionary_vectors = {}
    f = open('output/vector_space.txt','r')
    for line in f.readlines():
        features = line.split(",")
        n = len(features)
        vector = features[1:n-1]
        vector_float = [float(x) for x in vector]
        key_word = features[0]
        dictionary_vectors[key_word] = vector_float
    f.close()
    f_output = open(path, 'w')
    heading = get_heading()
    f_output.write(heading)
    the_vector = {}
    avg_vec_auto_yes_train = get_average_of_list(auto_yes_train, dictionary_vectors)
    avg_vec_auto_no_train = get_average_of_list(auto_no_train, dictionary_vectors)
    avg_vec_auto_yes_test = get_average_of_list(auto_yes_test, dictionary_vectors)
    avg_vec_auto_no_test = get_average_of_list(auto_no_test, dictionary_vectors)
    for i in range(len(list_of_classes)-1):
        line = text_list[i]
        the_class = list_of_classes[i]
        text = line #word_tokenize(line)
        f1 = automatic_freq(auto_yes_train,text)
        f2 = automatic_freq(auto_no_train,text)
        f3 = automatic_freq(auto_yes_test,text)
        f4 = automatic_freq(auto_no_test,text)
        word_embedding, numpy_word_embedding = get_word_embedding_128(dictionary_vectors, text)
        f5 = 0.0
        f6 = 0.0
        f7 = 0.0
        f8 = 0.0
        f9 = 0.0
        f10 = 0.0
        f5 = sim_tweet_to_avg_list(numpy_word_embedding,avg_vec_auto_yes_train)
        f6 = sim_tweet_to_avg_list(numpy_word_embedding,avg_vec_auto_no_train)
        f7 = sim_tweet_to_avg_list(numpy_word_embedding,avg_vec_auto_yes_test)
        f8 = sim_tweet_to_avg_list(numpy_word_embedding,avg_vec_auto_no_test)
        f9 = sim_avg_no_avg_yes(avg_vec_auto_yes_train, avg_vec_auto_no_train)
        f10 = sim_avg_no_avg_yes(avg_vec_auto_yes_test, avg_vec_auto_no_test)

        f_output.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,word_embedding,the_class))
    f_output.close()


##########################################################################################

def load_vector_space():
    path = 'output/vector_space.txt'
    df = pd.read_csv(path, sep=",")
    Vectors = df.ix[:,1:128]
    Words = df.ix[:,0]
    #print df.ix[1:2,1:4]
    #x = raw_input()
    numpyMatrix = Vectors.as_matrix()
    return numpyMatrix, Words

##########################################################################################


def get_results_list(indices_auto_train, similarity_df, Words, my_top_n):
    top_n = my_top_n
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
                    if word not in result_test_auto_similar:
                        result_test_auto_similar.append(word)

    return result_test_auto_similar

##########################################################################################

def get_results_list2(indices_auto_train, similarity_df, Words, my_top_n):
    top_n = my_top_n
    result_test_auto_similar = []
    dictionary_test_freq = {}
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
                    if word in dictionary_test_freq:
                        dictionary_test_freq[word] = dictionary_test_freq[word] + 1
                    else:
                        dictionary_test_freq[word] = 1

    return dictionary_test_freq

##########################################################################################
## better not to use m

def get_grams_deep_gramulator(auto_yes_train, auto_no_train, m, my_top_n):
    numpyMatrix, Words = load_vector_space()
    similarity = np.asarray(np.asmatrix(numpyMatrix) * np.asmatrix(numpyMatrix).T )
    similarity_df = pd.DataFrame(similarity)
    print pd.DataFrame(similarity, index=Words, columns=Words).ix[1:10,1:10]
    #indices_auto_yes_train = [int(i) for i in range(len(auto_yes_train)) if Words[i] in auto_yes_train]  ##correct
    #indices_auto_no_train = [int(i) for i in range(len(auto_no_train)) if Words[i] in auto_no_train]     ##correct
    ###############################################################
    indices_auto_yes_train = [int(i) for i in range(len(Words)) if Words[i] in auto_yes_train]
    indices_auto_no_train = [int(i) for i in range(len(Words)) if Words[i] in auto_no_train]
    ###############################################################
    #selected_rows_auto_yes_train = similarity_df.ix[indices_auto_yes_train]
    #selected_rows_auto_no_train = similarity_df.ix[indices_auto_no_train]
    result_test_auto_yes_similar = get_results_list(indices_auto_yes_train, similarity_df, Words, my_top_n)
    result_test_auto_no_similar = get_results_list(indices_auto_no_train,similarity_df, Words, my_top_n)
    print len(indices_auto_yes_train)
    print len(indices_auto_no_train)
    print len(Words)
    return result_test_auto_yes_similar, result_test_auto_no_similar  #[0:m]  ##reduced nos

##########################################################################################


def get_grams_deep_gramulator2(auto_yes_train, auto_no_train, m, my_top_n):
    numpyMatrix, Words = load_vector_space()
    similarity = np.asarray(np.asmatrix(numpyMatrix) * np.asmatrix(numpyMatrix).T )
    similarity_df = pd.DataFrame(similarity)
    print pd.DataFrame(similarity, index=Words, columns=Words).ix[1:10,1:10]
    ###############################################################
    indices_auto_yes_train = [int(i) for i in range(len(Words)) if Words[i] in auto_yes_train]
    indices_auto_no_train = [int(i) for i in range(len(Words)) if Words[i] in auto_no_train]
    ###############################################################
    dict_result_test_auto_yes_similar = get_results_list2(indices_auto_yes_train, similarity_df, Words, my_top_n)
    dict_result_test_auto_no_similar = get_results_list2(indices_auto_no_train,similarity_df, Words, my_top_n)
    
    words_dictionary_freq_no_sorted = []
    words_dictionary_freq_yes_sorted = []
    dictionary_freq_no_sorted = sorted(dict_result_test_auto_no_similar.items(), key=operator.itemgetter(1) , reverse=True)
    for tup in dictionary_freq_no_sorted:
        words_dictionary_freq_no_sorted.append(tup[0])
  
    dictionary_freq_yes_sorted = sorted(dict_result_test_auto_yes_similar.items(), key=operator.itemgetter(1) , reverse=True)
    for tup in dictionary_freq_yes_sorted:
        words_dictionary_freq_yes_sorted.append(tup[0])

    list_automatic_freq_yes = []
    list_automatic_freq_no = []


    for word in words_dictionary_freq_no_sorted[0:int(round(len(words_dictionary_freq_no_sorted)/2))]:
        if word not in words_dictionary_freq_yes_sorted[0:int(round(len(words_dictionary_freq_yes_sorted)/2))]:
            list_automatic_freq_no.append(word)
    
    for word in words_dictionary_freq_yes_sorted[0:int(round(len(words_dictionary_freq_yes_sorted)/2))]:
        if word not in words_dictionary_freq_no_sorted[0:int(round(len(words_dictionary_freq_no_sorted)/2))]:
            list_automatic_freq_yes.append(word)

    return list_automatic_freq_yes, list_automatic_freq_no



##########################################################################################
##########################################################################################
################################################################################
## Main()


auto_yes_train = []
auto_no_train = []
auto_yes_test = []
auto_no_test = []

path_train = 'input/PennData/train.txt'
path_test = 'input/PennData/test.txt'

list_of_texts_train_X, list_of_class_train_y = get_text_from_file(path_train)
list_of_texts_test_X, list_of_class_test_y = get_text_from_file(path_test)

auto_yes_train, auto_no_train = create_frequency_counts(list_of_texts_train_X, list_of_class_train_y , 11000)

## this func generates a text file which is input for the word2vec method
combined_train_test = combine_train_test_for_word2vec(list_of_texts_train_X, list_of_texts_test_X)


run_gensim(combined_train_test)


###########################################################################################################################

my_skip_window = 1 # How many words to consider left and right. usually 1
size_of_vector=128
my_num_steps=100001
my_vocabulary_size=10300
my_num_points=10000   #make this a little less than my_vocabulary_size
    
##########################################################################################

words = read_data2("output/combinedText.csv")

##########################################################################################

data_index = 0
vocabulary_size = my_vocabulary_size   ##correct entry
print('Data size %d' % len(words))

data, count, dictionary, reverse_dictionary = build_dataset(words)

print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

batch_size = size_of_vector
embedding_size = size_of_vector  # Dimension of the embedding vector.
skip_window = my_skip_window # 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.

valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32) ## this is a tensor
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                             labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

num_steps = my_num_steps  ## correct entry

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  average_loss = 0
  for step in range(num_steps):
    batch_data, batch_labels = generate_batch(
      batch_size, num_skips, skip_window)
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += l
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
        # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step %d: %f' % (step, average_loss))
      average_loss = 0
      # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
  final_embeddings = normalized_embeddings.eval()

num_points = my_num_points
print('data:', [reverse_dictionary[di] for di in data[:8]])
words = [reverse_dictionary[i] for i in range(1, num_points+1)]
write_vector_space_to_file(final_embeddings, words) ## ricardo added - 128 d vector file

########################################################################################################################

### auto_yes_test, auto_no_test = get_grams_deep_gramulator(auto_yes_train, auto_no_train, m=2000, my_top_n=60)
auto_yes_test, auto_no_test = get_grams_deep_gramulator2(auto_yes_train, auto_no_train, m=2000, my_top_n=45) ##deep freq version


path_out_train = 'output/train.txt'
path_out_test = 'output/test.txt'

create_feature_vectors(path_out_train,list_of_texts_train_X, list_of_class_train_y, auto_yes_train,auto_no_train,auto_yes_test, auto_no_test)
create_feature_vectors(path_out_test, list_of_texts_test_X, list_of_class_test_y, auto_yes_train,auto_no_train,auto_yes_test, auto_no_test)


############################################################################


print '<<<<<<<<<<<<<DONE>>>>>>>>>>>'
