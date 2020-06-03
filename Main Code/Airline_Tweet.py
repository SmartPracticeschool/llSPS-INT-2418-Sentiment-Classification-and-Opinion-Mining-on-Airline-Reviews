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



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps=PorterStemmer()
data=[]


dataset=pd.read_csv(r"Tweets.csv",encoding="ISO-8859-1")

#GRAPHS
# Group by airline_sentiment
dataset_processed = dataset.groupby("airline_sentiment")["airline_sentiment"].count().to_dict()

# Add on the values of y-axis into the plot
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height, '%f' % float(height),
                ha='center', va='bottom')

# Bar Chart
fig, ax = plt.subplots()
index = list(dataset_processed.keys())
_bar=ax.bar(index, list(dataset_processed.values()), color=['r','b','g','y'])
plt.xticks(index, list(dataset_processed.keys()), rotation=0)
plt.xlabel('Types of Sentiments')
plt.ylabel('Count of Sentiments')
plt.title('The Count versus Type of Sentiments')
autolabel(_bar)
plt.show()



# Group by airline_sentiment and airline
dataset_processed = dataset.groupby(["airline", "airline_sentiment"]).count().iloc[:,0].unstack(0)

# Bar Chart
chart = dataset_processed.plot(kind='bar',title = 'The Count of Sentiments versus Airlines grouped by Types of Sentiments')
chart.set_xlabel('Types of Sentiments')
chart.set_ylabel('Count of Sentiments')
plt.show()


# Group by negativereason
dataset_processed = dataset.groupby("negativereason")["negativereason"].count().to_dict()



# Pie chart, where the slices will be ordered and plotted counter-clockwise:
fig, ax = plt.subplots()
ax.pie(list(dataset_processed.values()), labels=list(dataset_processed.keys()), autopct='%1.1f%%', shadow=True, startangle=0)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('The distribution of Sentiments among All Negative Reasons')
plt.show()



#Data Preprocessing
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


#Encoding the sentiment values for the model
x=dataset.iloc[:,:].values
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset['airline_sentiment']=le.fit_transform(dataset['airline_sentiment'])
from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder()
z=one.fit_transform(x[:,0:1]).toarray()
x=np.delete(x,0,axis=1)
x=np.concatenate((z,x),axis=1)


#Using Count Vectorizer to transform the data into suitable format(Numeric)
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=2000)
x=cv.fit_transform(data).toarray()
y=dataset['airline_sentiment'].values
x.shape
y.shape


#Splitting into train and test values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y, random_state=0, test_size=0.2)



#Applying Convolution Neural Network and training the model
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units = 2000 ,init = "random_uniform",activation = "relu"))
model.add(Dense(units = 4000 ,init = "random_uniform",activation = "relu"))

model.add(Dense(units = 3 ,init = "random_uniform",activation = "softmax"))
model.compile(optimizer = "adam",loss = "sparse_categorical_crossentropy",metrics = ["accuracy"])
model.fit(x_train,y_train,epochs  = 20)



#Testing our model with a random input
text =  "VirginAmerica @freddieawards Done and done! Best airline around, hands down!"
text = re.sub('[^a-zA-Z]', ' ',text)
text = text.lower()
text = text.split()
text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
text = ' '.join(text)
y_p1 = model.predict_classes(cv.transform([text]))
if(y_p1==0):
        y_p1="Negative"
elif(y_p1==1):
    y_p1="Neutral"
else:
    y_p1="Positive"
y_p1


#Saving the trained model for application

import pickle
pickle.dump(model,open('Airline_Tweet model.pkl','wb')) #Saving model
pickle.dump(cv,open('Airline_Tweet cv.pkl','wb')) #Saving Count Vectorizer











