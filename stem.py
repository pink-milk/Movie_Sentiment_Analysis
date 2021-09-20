# from __future__ import print_function
from nltk.stem import PorterStemmer
import numpy as np
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
# nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english') 

stop_words = stopwords.words('english')

# print(stop_words)
#remove stop words
#stopwords.words('english')

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    punc = re.sub(r'[^\w\s]','',cleantext)

    return punc


stemmer = SnowballStemmer("english")


with open("train_test.txt", "r") as train_data:
	lines = train_data.readlines()
   
# lines=stemmer.stem(lines)

# print(lines)


rating = []
review = []

# new_text = ""
# for word in words:
#     if word not in stop_words:
#         new_text = new_text + " " + word

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
        if w not in stop_words:
            x = stemmer.stem(w)
            stemmed.append(x)
        # else:
        #     print(w)
    # print(stemmed) 
    review.append(stemmed)
    
    # print(words)

# print(stop_words)
# print(rating[2])
print(review[1])
