# from __future__ import print_function
from nltk.stem import PorterStemmer
import numpy as np
import re
from nltk.stem.snowball import SnowballStemmer

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    punc = re.sub(r'[^\w\s]','',cleantext)

    return punc


stemmer = SnowballStemmer("english")
print(stemmer.stem("running away"))


with open("train_test.txt", "r") as train_data:
	lines = train_data.readlines()
   
# lines=stemmer.stem(lines)

# print(lines)


rating = []
review = []

for l in lines:
    #put ratings, reviews in different lists, splice at first tab
    as_list = l.split("\t",1)
    
    rating.append(as_list[0])
    # as_list[1]=stemmer.stem(as_list[1])

    #clean html in review
    cleaned=(cleanhtml(as_list[1]))

    #splice review into [] of words
    words=cleaned.split()

    #stem the words in words[] 
    stemmed=[]
    for w in words:
        x = stemmer.stem(w)
        stemmed.append(x)

    review.append(stemmed)
    # print(words)


print(rating[2])
print(review[2])
