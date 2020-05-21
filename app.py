# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:29:19 2019

@author: lalit
"""
    
from flask import render_template, Flask, request,url_for
from keras.models import load_model
import pickle 
with open(r'CountVectorizer','rb') as file:
    cv=pickle.load(file)
cla = load_model('toxic_comments.h5')
cla.compile(optimizer='adam',loss='binary_crossentropy')

app = Flask(__name__)

@app.route('/')


def home():
    return render_template('index.html')
@app.route('/tpredict')


@app.route('/login', methods = ['POST'])

def login():
   
    if request.method == 'POST':
        topic = request.form['tweet']
        print("Hey " +topic)
        topic=cv.transform([topic])
        print("\n"+str(topic.shape)+"\n")
        y_pred = cla.predict(topic)
        print("pred is "+str(y_pred))
        topic = list(y_pred)
        return render_template('index.html',showcase = topic)


if __name__ == '__main__':
    app.run(host = 'localhost', debug = True , threaded = False)
    
