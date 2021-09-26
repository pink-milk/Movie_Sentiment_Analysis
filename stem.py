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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

stop_words = stopwords.words('english')

# print(stop_words)
#remove stop words
#stopwords.words('english')

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    punc = re.sub(r'[^\w\s]','',cleantext)

    return punc

# def vectorize(processed_text):
#     DF = {}
#     for i in range(len(processed_text)):
#         tokens = processed_text[i]
#         # for w in tokens:
#         try:
#             DF[processed_text[i]]+=1
#         except:
#             DF[processed_text[i]] = 1
#             # print('err 34')
#     processed_text[i]
#     print('unique words: '+str(len(DF)) )
#     return DF

# def fit_corpus(train_data, test_data):
#     corpus = pd.DataFrame({"review": train_data["review"]})
#     corpus.reviews.append(test_data["review"], ignore_index=True)
#     tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))
#     tfidf.fit(corpus["review"])
#     return tfidf

stemmer = SnowballStemmer("english")


with open("train_data.txt", "r") as train_data:
	lines = train_data.readlines()
with open("test_file.txt", "r") as test_data:
	lines2 = test_data.readlines()   
# lines=stemmer.stem(lines)

# print(lines)


rating = []
review = []

rating2=[]
review2=[]

# for l2 in lines2:
#     #put ratings, reviews in different lists, splice at first tab
#     as_list = l2.split("\t",1)
    
#     rating2.append(as_list[0])
#     # as_list[1]=stemmer.stem(as_list[1])

#     #clean html in review
#     cleaned=(cleanhtml(as_list[1]))

#     #splice review into [] of words
#     words=cleaned.split()

#     #stem the words in words[] 
#     stemmed=[]
     
#     for w in words:
#         if w not in stop_words:
#             x = stemmer.stem(w)
#             stemmed.append(x)
        
#     # print(stemmed) 
#     review2.append(stemmed)
    
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
     
    for w in l:
        if w not in stop_words:
            x = stemmer.stem(w)
            stemmed.append(x)
        
    review.append(stemmed)
    
    # print(words)

# print(stop_words)
# print(rating)
# print(review[5])

df = pd.DataFrame({"review": review, "label": rating})

# pos2 = pd.DataFrame({"review": review2, "label": rating2})



# vectorizer = TfidfVectorizer (max_features=2500, min_df=2, max_df=0.8, stop_words=stopwords.words('english'))
# processed_features = vectorizer.fit_transform(review[3])
# print(processed_features)
# print(len(processed_features))

# print(df['label'])
# print(vectorize(review[3]))


# tfidf = fit_corpus(pos, pos2)
vectorizer= TfidfVectorizer() 

x= vectorizer.fit_transform(lines)
y= df.label

x2= vectorizer.transform(lines2)
# y2= df.label2

# print(type(x2))
# print(lines[1])
cos=cosine_similarity(x2,x)

# print(cos.shape)
# print(type(cos[1]))
# print(cos)

# cos.sort()

# print(cos)

res=[]

#sort the cos simularity matrix
for i in range(len(cos)):
    arr=(list(enumerate((cos[i]))))
    arr.sort(key=lambda x:x[1])
    res.append(arr)
    # print(arr)
# print(type(res))

#get k nearest neighbors
def get_knn(sorted_sim, k):
    k_matrix=[]
    for i in range(len(sorted_sim[1])):
        k_matrix.append(sorted_sim[i][-k:])

    return k_matrix
# print(res)
knn=get_knn(res,25)
# print(knn)
# print(knn[1][1][0])
# print(rating[5])

#get the knn tuple matrix of k most similar reviews, get the indexes of the k reviews,
#-then get the sentiments from rating arr, sum k sentiments, 
#returns sum arr (if positive, most likely a positive review) for each test review
def get_sentiments_knn(knn_tuples, rating_arr):
    
    numsarr=[]
    for i in range(len(knn)):
        nums=[]
        for j in range(len(knn[i])):
            index=knn[i][j][0]
            #grab the rating, turn it into an int
            nums.append(int(rating[index]))
        a=sum(nums)
        numsarr.append(a)

    return numsarr    
        
# print(get_sentiments_knn(knn, rating))

add=[]
# for i in range(len(knn[1])):
#     k=3
#     each=[]
#     while(k>=0)
#         print(knn[i][k])

senti=get_sentiments_knn(knn, rating)


with open('output2.txt', 'w') as f:
    for item in senti:
        if(item>0):
            f.write("+1\n")
        else:
            f.write("-1\n")


# print(res)
x_train,x_test,y_train,y_test = train_test_split(x,rating,test_size=0.2)

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(x)


# print(model_knn)

# classifier = RandomForestClassifier(n_estimators=100,)
# classifier.fit(x_train,y_train)

#making predictions
# y_pred = classifier.predict(x_test)#model accuracy
# print("Model Accuracy : {}%".format((y_pred == y_test).mean()))#confusion matrix 
# print(confusion_matrix(y_test,y_pred))
# print(x)
# vectorizer= TfidfVectorizer()
# tf_x_train = vectorizer.fit_transform(pos)
# tf_x_test = vectorizer.transform(pos2)

# print(tf_x_train)
# print(tf_x_test)