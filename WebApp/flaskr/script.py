import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

app=Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,19)
    loaded_model = pickle.load(open("rfc_model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        #to_predict_list = request.form.to_dict()
        #to_predict_list=list(to_predict_list.values())
        #to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result)>0.5:
            prediction='He subscribes'
        else:
            prediction='He doesnt subscribe.'
        return render_template("result.html",prediction=result)