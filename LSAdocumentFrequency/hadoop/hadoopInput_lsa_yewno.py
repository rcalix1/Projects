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
import os
import math

###############################################################################

list_of_texts = []
list_of_titles = []
list_of_authors = []

#############################################################################

def get_length_of_doc(file):
    f = open(file)
    contents = f.read()
    temp = contents.split(" ")
    return len(temp)

#############################################################################

df = pd.read_csv("/home/rcalix/Desktop/hadoopCODEtfidf/yewno-output",sep='\t',names = ["term", "file", "tf", "df"])
df['file'] = df['file'].str.replace('hdfs://localhost:54310/user/hduser/', '')
df['file'] = df['file'].str.replace('ricardo-output2/','')
df['term'] = df['term'].apply(lambda x: x.split('\thdfs://')[0])
df = df[df.file != 'part-00000']
columns_terms = df.term.unique()
rows_files = df.file.unique()
df['tfidf'] = df.tf * df.df
#df = df.groupby(["term","file"]).sum()
#print df.ix[1:20,'term']
#df = df.pivot(index='file', columns='term', values='tf')
table = pd.pivot_table(df, values='tfidf', rows=['file'], cols=['term'], aggfunc=np.sum)
table = table.replace("NaN",0)
table_dtm = table

#print df
print table_dtm
#rr = raw_input()

##############################################################################

#vectorizer = CountVectorizer(min_df=1, stop_words='english')
#vec = vectorizer.fit(list_of_texts)
#dtm = vec.transform(list_of_texts)

############################################################################

#tfidf = TfidfTransformer()
#tfidfMatrix_metric = tfidf.fit(dtm)
#tfidfMatrix  = tfidfMatrix_metric.transform(dtm)

#print pd.DataFrame(tfidfMatrix.toarray(), index=list_of_authors, columns=vec.get_feature_names()).head(44)
#rr= raw_input()

#############################################################################

#lsa = TruncatedSVD(2, algorithm='arpack') ## 2 components
lsa = TruncatedSVD(36, algorithm='arpack') ## 40 components
model = lsa.fit(table_dtm)
dtm_lsa = model.transform(table_dtm)

##########################################################################

Norm_obj = Normalizer(copy=False).fit(dtm_lsa)
dtm_lsa = Norm_obj.transform(dtm_lsa)

###########################################################################
## plot

def plot_2d_data():
    xs = [w[0] for w in dtm_lsa]
    ys = [w[1] for w in dtm_lsa]

    fig = plt.figure()
    size_list = len(rows_files)
    for i in range(size_list):
        plt.scatter(xs[i], ys[i])

    for i in range(size_list):
        plt.annotate(rows_files[i], xy = (xs[i], ys[i]), xytext = (20, 20), textcoords = 'offset points', 
                      size=15, ha = 'right', va = 'bottom')
    plt.xlabel('first principal component')
    plt.ylabel('second principal component')
    plt.title('plot of docs and authors against lsa components')
    plt.show()

############################################################################

plot_2d_data()


##########################################################################
## similarity


similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T )
#print similarity
print pd.DataFrame(similarity, index=rows_files, columns=rows_files)

#################################################################


print '<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>'


