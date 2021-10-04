# from __future__ import print_function
from nltk.stem import PorterStemmer
import numpy as np
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def cleanhtml(raw_html):
    #remove html tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    #remove punctuation
    punc = re.sub(r'[^\w\s]','',cleantext)
    #lowercase
    lowe=punc.lower()
    return lowe

stop_words = stopwords.words('english')
stemmer = SnowballStemmer("english")

with open("train_data.txt", "r") as train_data:
    lines = train_data.readlines()[1:]
    
with open("test_file.txt", "r") as test_data:
	lines2 = test_data.readlines()   

rating = []

new_corpus2=[]
for l2 in lines2:
    #clean html in review
    cleaned=(cleanhtml(l2))
    #splice review into [] of words
    words=cleaned.split()
    new_text=""
    for w in words:
        if w not in stop_words:
            x = lemmatizer.lemmatize(w)
            new_text=new_text+" "+x
    new_corpus2.append(new_text)
# print(new_corpus2)

new_corpus=[]
# print(lines)
for l in lines:
    #put ratings, reviews in different lists, splice at first tab
    as_list = l.split("\t",1)
    rating.append(as_list[0])
    # as_list[1]=stemmer.stem(as_list[1])
    
    #clean html in review
    cleaned=(cleanhtml(as_list[1]))

    #splice review into [] of words
    words=cleaned.split()
    # print(cleaned)

    #stem the words in words[] and put into new corpus
    new_text=""

    for w in words:
        
        if w not in stop_words:
            x = lemmatizer.lemmatize(w)
            new_text=new_text+" "+x
            # stemmed.append(x)
    # print(new_text)
    new_corpus.append(new_text)
    
# vectorize the corpus
vectorizer= TfidfVectorizer() 

x= vectorizer.fit_transform(new_corpus)
feature_names = vectorizer.get_feature_names()
print(feature_names)
# y= df.label
x2= vectorizer.transform(new_corpus2)
# y2= df.label2
cos=cosine_similarity(x2,x)

#sort the cos simularity matrix
#args: cossimularity matrix
#returns matrix of tuples
def sort_and_tuplize(cos):
    res=[]
    for i in range(len(cos)):
        arr=(list(enumerate((cos[i]))))
        arr.sort(key=lambda x:x[1])
        res.append(arr)
        # print(arr)
    # print(type(res))
    return res

#get k nearest neighbors, 
#args: sorted simularity matrix, k
def get_knn(sorted_sim, k):
    k_matrix=[]
    for i in range(len(sorted_sim[1])):
        k_matrix.append(sorted_sim[i][-k:])

    return k_matrix

#get the knn tuple matrix of k most similar reviews, get the indexes of the k reviews,
#-then get the sentiments from rating arr, sum k sentiments, 
#returns sum arr (if positive, most likely a positive review) for each test review
def get_sentiments_knn(knn_tuples, rating_arr):
    numsarr=[]
    for i in range(len(knn)):
        nums=[]
        
        for j in range(1,len(knn[i])):
            index=knn[i][j][0]
            #grab the rating, turn it into an int
            nums.append(int(rating[index]))
        a=sum(nums)
        numsarr.append(a)

    return numsarr   

res=sort_and_tuplize(cos)
knn=get_knn(res,75)
senti=get_sentiments_knn(knn, rating)

#write -1, +1 classifications to output file
with open('output9.txt', 'w') as f:
    for item in senti:
        if(item>0):
            f.write("+1\n")
        else:
            f.write("-1\n")

