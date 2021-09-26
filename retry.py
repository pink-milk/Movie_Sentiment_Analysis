# from __future__ import print_function
from nltk.stem import PorterStemmer
import numpy as np
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
# nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# temp = []
with open("train_test.txt", "r") as train_data:
    temp=train_data.readlines()
print(temp)


rating=[]
as_list=[]
for l in temp:
    as_list = l.split("\t",1)
    rating.append(as_list[0])

print(rating)


print(pos)