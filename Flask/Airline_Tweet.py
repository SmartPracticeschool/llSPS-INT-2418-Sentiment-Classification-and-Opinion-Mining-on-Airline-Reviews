#Optimizations as per CPU
from keras import backend as K
import tensorflow as tf
import os
NUM_PARALLEL_EXEC_UNITS = 8
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                       allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.Session(config=config)
K.set_session(session)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"


import numpy as np
import pandas as pd
dataset=pd.read_csv(r"Tweets.csv",encoding="ISO-8859-1")
# dataset=dataset[['airline_sentiment','text']]
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
data=[]
x=dataset.iloc[:,:].values


from textblob import TextBlob
#$ pip install -U textblob
#$ python -m textblob.download_corpora
from nltk.stem.wordnet import WordNetLemmatizer
def text_processing(tweet):

    #Generating the list of words in the tweet (hastags and other punctuations removed)
    def form_sentence(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)
    new_tweet = form_sentence(tweet)

    #Removing stopwords and words with unusual symbols
    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess
    no_punc_tweet = no_user_alpha(new_tweet)

    #Normalizing the words in tweets
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet


    return normalization(no_punc_tweet)

dataset['tweet_list'] = dataset['text'].apply(text_processing)
dataset[dataset['airline_sentiment']==1].drop('text',axis=1).head()

for i in range(0,14640):
    review = dataset[['tweet_list'][0]][i]
    review = ' '.join(review)
    data.append(review)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset['airline_sentiment']=le.fit_transform(dataset['airline_sentiment'])
from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder()
z=one.fit_transform(x[:,0:1]).toarray()
x=np.delete(x,0,axis=1)
x=np.concatenate((z,x),axis=1)



from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=2000)
x=cv.fit_transform(data).toarray()
y=dataset['airline_sentiment'].values
x.shape
y.shape



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y, random_state=0, test_size=0.2)
# x_train.values
# y_train.values

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units = 2000 ,init = "random_uniform",activation = "relu"))
model.add(Dense(units = 4000 ,init = "random_uniform",activation = "relu"))

model.add(Dense(units = 3 ,init = "random_uniform",activation = "softmax"))
model.compile(optimizer = "adam",loss = "sparse_categorical_crossentropy",metrics = ["accuracy"])
model.fit(x_train,y_train,epochs  = 20)


text =  "@VirginAmerica did you know that suicide is the second leading cause of death among teens 10-24"
text = re.sub('[^a-zA-Z]', ' ',text)
text = text.lower()
text = text.split()
text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
text = ' '.join(text)

y_p1 = model.predict_classes(cv.transform([text]))

y_p1

import pickle
pickle.dump(model,open('Airline_Tweet model.pkl','wb'))
pickle.dump(cv,open('Airline_Tweet cv.pkl','wb'))











