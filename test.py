# from __future__ import print_function
# from nltk.stem import *

# from nltk.stem.snowball import SnowballStemmer

import numpy as np
import re


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    punc = re.sub(r'[^\w\s]','',cleantext)

    return punc


# stemmer = SnowballStemmer("english")
# print(stemmer.stem("running"))

with open("train_data.txt", "r") as train_data:
	lines = train_data.readlines()

# print lines


rating = []
review = []

for l in lines:
    as_list = l.split("\t",1)
    rating.append(as_list[0])
    review.append(cleanhtml(as_list[1]))
    # review.append(as_list[1])

# print(rating)
print(review[1])