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


################################################
## read data into a string


def read_data2(filename):
    f = open(filename)
    data = tf.compat.as_str(f.read()).split()
    return data

#################################################

#words = read_data2("ricardo.txt")
#words = read_data2("/home/rcalix/Desktop/datamatrika/results/All_217741_cleaner.txt")
words = read_data2("/home/rcalix/Desktop/DeepGramulator/word2vec/preprocessing/output/word2vec_input_12559_3156_combined.csv")


#print words
print('Data size %d' % len(words))

#x = raw_input()

#######################################################

###########################################################
words_to_plot = []
def func_words_to_plot():
    f_plot = open("words_to_plot.txt","r")
    for word in f_plot.readlines():
        words_to_plot.append(word.replace("\n",""))

    f_plot.close()

#############################################################

#func_words_to_plot()

#print words_to_plot


#######################################################

vocabulary_size = 26938 ##correct entry
#vocabulary_size =  4090 ## ricardo

def build_dataset(words):
  count = [['UNK', -1]]
  print count
  #x = raw_input()
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  #for i in range(100):
  #print count[0:30]
  #x = raw_input()
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
    #print dictionary[word]
    #x = raw_input()
  #print dictionary['took']
  #x = raw_input()
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
  #print data
  #x = raw_input()
  #print unk_count
  #x = raw_input()
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  #x = raw_input()
  #print reverse_dictionary[7]
  #x = raw_input()
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.
#x = raw_input()
######################################################

##function to generate a training batch for the skip-gram model

data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
  global data_index
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
      #print "#####################################################################"
      #print "buffer ", buffer
      #print "targets to avoid ", targets_to_avoid
      #print "batch ", batch
      #print "labels ", labels
      #print "buffer ", buffer
      #print "data", data[:20]
      #x = raw_input()

    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])
#x = raw_input()
# marcela here

for num_skips, skip_window in [(2, 1), (4, 2)]:
    #print num_skips
    #print skip_window
    #x = raw_input()
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    #print "batch", batch
    #x = raw_input()
    #print "labels", labels
    #print "data", data[:20]
    #x = raw_input()

    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    #x = raw_input()
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
    #x = raw_input()

#print "batch", batch
#print "labels", labels
#print "data", data[:20]
#x = raw_input()

#######################################################
## Train a skip gram model

batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. 
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):

  # Input data.
  train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  #print valid_examples
  
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32) ## this is a tensor
  #print "valid data set", valid_dataset
  #x = raw_input()
  # Variables.
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  #print embeddings
  #x = raw_input()
  softmax_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                         stddev=1.0 / math.sqrt(embedding_size)))
  softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Model.
  # Look up embeddings for inputs.
  embed = tf.nn.embedding_lookup(embeddings, train_dataset)
  # Compute the softmax loss, using a sample of the negative labels each time.
  loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                               train_labels, num_sampled, vocabulary_size))

  # Optimizer.
  # Note: The optimizer will optimize the softmax_weights AND the embeddings.
  # This is because the embeddings are defined as a variable quantity and the
  # optimizer's `minimize` method will by default modify all variable quantities 
  # that contribute to the tensor it is passed.
  # See docs on `tf.train.Optimizer.minimize()` for more details.
  optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
  
  # Compute the similarity between minibatch examples and all embeddings.
  # We use the cosine distance:
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

#######################################################

num_steps = 100001 ## correct entry
#num_steps = 30001  ## ricardo added

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

#print "ricardo"
#print final_embeddings
#x = raw_input()
######################################################

#print final_embeddings
#print "did you see an embedding?"
#x = raw_input()
#f_space = open("word2vec_vector_space","w")
#for item in final_embeddings:
#    f_space.write(item)
#f_space.close()
def write_vector_space_to_file(embeddings, labels):
    #pickle.dump(final_embeddings, open("word2vec_vector_space", "w"))
  f_vector_space = open("output/deep_gramulator_word2vec_vector_space.txt","w")
  features = []
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  #pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    features = embeddings[i,:]
    #pylab.scatter(x, y)
    features_string = ""
    for feat in features:
        features_string = features_string + str(feat) + ","
    string = str(label) + "," + features_string
    f_vector_space.write(string)
    f_vector_space.write("\n")
  f_vector_space.close()   
   

########################################################################################

print "num points ricardo"
print len(reverse_dictionary)

num_points = 20000 ## correct entry ## points (i.e. words) to display on the plot 
#num_points = 98 ## ricardo added

#tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])  

########################################################################################


def word_present_in_lists(word, yes_train, yes_test, no_train, no_test):
    presence_string = ""
    a = 0
    b = 0
    c = 0
    d = 0
    if word in yes_train:
        a = 1
    if word in yes_test:
        b = 1
    if word in no_train:
        c = 1
    if word in no_test:
        d = 1
    presence_string = str(a) + str(b) + str(c) + str(d)
    return presence_string


########################################################################################

def plot(embeddings, labels, yes_train, yes_test, no_train, no_test):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  already_seen_words_in_train = []
  for i, label in enumerate(labels):
    #presence_string = word_present_in_lists(label,yes_train, yes_test, no_train, no_test)
    #print presence_string
    #rr = raw_input()
    x, y = embeddings[i,:]
    #pylab.scatter(x, y)
    label = label.lower()
    if label in yes_train:
      pylab.scatter(x,y, color='blue')
      pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
      already_seen_words_in_train.append(label)
#    if label in no_train:
#      pylab.scatter(x,y, color='red')
#      already_seen_words_in_train.append(label)

  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    label = label.lower()
    if label not in already_seen_words_in_train:
      if label in yes_test:
        pylab.scatter(x,y, color='green')
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
#      if label in no_test:
#        pylab.scatter(x,y, color='yellow')
        #pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

  pylab.show()

#######################################################################################

def gramulator_list(file_name):
    list_of_grams = []
    f_grams = open(file_name)
    for word in f_grams.readlines():
        list_of_grams.append(word.replace("\n",""))
    f_grams.close()
    return list_of_grams

######################################################################################

path_grams = '/home/rcalix/Desktop/DeepGramulator/gramulator_features/'

yes_train = gramulator_list(path_grams + 'train_auto_yes_gramulator_file.txt')
yes_test = gramulator_list(path_grams + 'test_auto_yes_gramulator_file.txt')
no_train = gramulator_list(path_grams + 'train_auto_no_gramulator_file.txt')
no_test = gramulator_list(path_grams + 'test_auto_no_gramulator_file.txt')

#print yes_train
#rr=raw_input()
#print yes_test
#rr= raw_input()
#print no_train
#rr = raw_input()
#print no_test
#rr = raw_input()

#######################################################################################

words = [reverse_dictionary[i] for i in range(1, num_points+1)]
#plot(two_d_embeddings, words, yes_train, yes_test, no_train, no_test) 
write_vector_space_to_file(final_embeddings, words) ## ricardo added - 128 d vector file
#write_vector_space_to_file(two_d_embeddings, words) ## ricardo added - 2d vector file



#######################################################

print '<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>'
