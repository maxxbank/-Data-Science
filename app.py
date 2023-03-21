# framework for web libraries
import flask
from flask import Flask
from flask import render_template #отрисовка html шаблонов
from flask import request #выполнение запросов

# Load Tensorflow Keras libraries
import tensorflow as tf

# Load damps and loads libraries
import pickle # damps and loads data

# Load the Sklearn libraries
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# app.py
app = flask.Flask(__name__, template_folder = 'templates') #хранение HTML шаблоном в директории

@app.route('/') #декораторы на главной странице

def select_model():
    return render_template('main.html') #отрисовка html main

def upr_prediction(parameters):
    with open('kneighbors_regressor_upr_best.pkl', 'rb') as pickle_file: # Загрузка модели в формате pickle
         reconstructed_model = pickle.load(pickle_file)
    pred = reconstructed_model.predict([parameters])
    return pred

def pr_prediction(parameters):
    with open('randomforest_regressor_pr_best.pkl', 'rb') as pickle_file: # Загрузка модели в формате pickle
         reconstructed_model = pickle.load(pickle_file)
    pred = reconstructed_model.predict([parameters])
    return pred

def mn_prediction(parameters):
    reconstructed_model = tf.keras.models.load_model('neural_mn_model') # Загрузка модели в формате Tensorflow
    pred = reconstructed_model.predict([parameters])
    return pred

@app.route('/upr/', methods = ['POST', 'GET']) #декораторы на странице upr
def upr_predict():
  message = ''
  if request.method == 'POST':
    parameters_list = ('mn', 'plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'pr', 'ps', 'yn', 'shn', 'pln')
    parameters = []
    for i in parameters_list:
            parameter = request.form.get(i) # передаем данные с HTML форм
            parameters.append(parameter)
    parameters = [float(i.replace(',', '.')) for i in parameters]
    message = f'Значение Модуля упругости при растяжении: {upr_prediction(parameters)} ГПа'
  return render_template('upr.html', message=message)

@app.route('/pr/', methods = ['POST', 'GET']) #декораторы на странице pr
def pr_predict():
  message = ''
  if request.method == 'POST':
    parameters_list = ('mn', 'plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'upr', 'ps', 'yn', 'shn', 'pln')
    parameters = []
    for i in parameters_list:
            parameter = request.form.get(i) # передаем данные с HTML форм
            parameters.append(parameter)
    parameters = [float(i.replace(',', '.')) for i in parameters]
    message = f'Значение Прочности при растяжении: {pr_prediction(parameters)} МПа'
  return render_template('pr.html', message=message)

@app.route('/mn/', methods = ['POST', 'GET']) #декораторы на странице upr
def mn_predict():
  message = ''
  if request.method == 'POST':
    parameters_list = ('plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'upr', 'pr', 'ps', 'yn', 'shn', 'pln')
    parameters = []
    for i in parameters_list:
            parameter = request.form.get(i) # передаем данные с HTML форм
            parameters.append(parameter)
    parameters = [float(i.replace(',', '.')) for i in parameters]
    message = f'Значение Cоотношения матрицы-наполнителя: {mn_prediction(parameters)}'
  return render_template('mn.html', message=message)

if __name__ == '__main__':
    app.run()