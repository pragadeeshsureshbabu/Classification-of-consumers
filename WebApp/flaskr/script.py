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
    result = loaded_model.predict_proba(to_predict)[:, 1]
    return result[0]
def ValuePredictor_log(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,19)
    loaded_model = pickle.load(open("log_model.pkl","rb"))
    result = loaded_model.predict_proba(to_predict)[:, 1]
    return result[0]
@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result1 = ValuePredictor(to_predict_list)
        result2 = ValuePredictor_log(to_predict_list)
        result=(result1+result2)/2.0
        f=result*100
        f=round(f,4)
        if float(result)>0.53:
            prediction='Yes, This customer will Subscribe with a probability of '+ str(f)+'%'
        else:
            ft=100-f
            ft=round(ft,4)
            prediction='No, This customer will Not Subscribe with a probability of '+ str(ft)+'%'
        return render_template("result.html",prediction=prediction)