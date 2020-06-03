from flask import Flask , render_template ,request
import pickle
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer
cv= pickle.load(open('Airline_Tweet cv.pkl','rb'))
app  = Flask(__name__)
model = pickle.load(open('Airline_Tweet model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template('basic.html')
@app.route('/login', methods = ["POST"])
def login():
    text = request.form["message"]
    text = re.sub('[^a-zA-Z]', ' ',text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    p = model.predict_classes(cv.transform([text]))
    if(p==0):
        p="Negative"
    elif(p==1):
        p="Neutral"
    else:
        p="Positive"
    return render_template('basic.html',label = "The sentiment is = "+p)
if __name__=='__main__':
    app.run(debug = True)