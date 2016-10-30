from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_etexts
from gutenberg.query import get_metadata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd

###############################################################################

list_of_texts = []
list_of_titles = []
list_of_authors = []


##############################################################################

for i in range(2700,2800):
    try:
        text = strip_headers(load_etext(i)).strip()
        lines = text.split('\n')
        titleandauthor = " ".join(lines[0:20])
        content = lines[20:]
        #rr = raw_input()
        main_text = "\n".join(content)
        if "By " in titleandauthor:
            head = titleandauthor.split("By ")
            title_string = head[0]
            temp = head[1]
            author_string = temp[:20]
            print '*************************************************************************'
            print title_string
            print author_string
            #rr = raw_input()
            list_of_titles.append(title_string)
            list_of_authors.append(author_string)
            list_of_texts.append(main_text)
            #f_out = open('corpus/' + str(i) + '.txt', 'w')
            #f_out.write(main_text)
            #f_out.close()
    except:
        print "cannot download"

############################################################################

print len(list_of_titles)

############################################################################

vectorizer = CountVectorizer(min_df=1, stop_words='english')
vec = vectorizer.fit(list_of_texts)
dtm = vec.transform(list_of_texts)

############################################################################

tfidf = TfidfTransformer()
tfidfMatrix_metric = tfidf.fit(dtm)
tfidfMatrix  = tfidfMatrix_metric.transform(dtm)


#############################################################################

#lsa = TruncatedSVD(2, algorithm='arpack') ## 2 components
lsa = TruncatedSVD(40, algorithm='arpack') ## 40 components
model = lsa.fit(tfidfMatrix)
dtm_lsa = model.transform(tfidfMatrix)

##########################################################################

Norm_obj = Normalizer(copy=False).fit(dtm_lsa)
dtm_lsa = Norm_obj.transform(dtm_lsa)

###########################################################################
## plot

def plot_2d_data():
    xs = [w[0] for w in dtm_lsa]
    ys = [w[1] for w in dtm_lsa]

    fig = plt.figure()
    size_list = len(list_of_authors)
    for i in range(size_list):
        plt.scatter(xs[i], ys[i])

    for i in range(size_list):
        plt.annotate(list_of_authors[i], xy = (xs[i], ys[i]), xytext = (20, 20), textcoords = 'offset points', 
                      size=15, ha = 'right', va = 'bottom')
    plt.xlabel('first principal component')
    plt.ylabel('second principal component')
    plt.title('plot of docs and authors against lsa components')
    plt.show()

############################################################################

plot_2d_data()

###########################################################################
## read in test sample

f = open("test_moby_dic.txt")
string_test = f.read()
samples_to_compare = []
authors_to_compare = []
samples_to_compare.append(string_test)
authors_to_compare.append("Melville")
print samples_to_compare


##########################################################################
## similarity

test_sample = vec.transform(samples_to_compare)
tfidf_test_sample  = tfidfMatrix_metric.transform(test_sample)
dtm_lsa_test_sample = model.transform(tfidf_test_sample)
dtm_lsa_test_sample = Norm_obj.transform(dtm_lsa_test_sample)

##

similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa_test_sample).T )
#print similarity
print pd.DataFrame(similarity, index=list_of_authors, columns=authors_to_compare)

############################################################################
#rr =raw_input()
#print(text)
#print(get_metadata('title', 2701))  # prints frozenset([u'Moby Dick; Or, The Whale'])
#print(get_metadata('author', 2701)) # prints frozenset([u'Melville, Hermann'])

#################################################################


print '<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>'


