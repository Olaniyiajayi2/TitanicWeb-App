from flask import Flask, render_template, url_for, request, session, redirect
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib



app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template("index.html")
	
	
@app.route('/preview')
def preview():
	df1 = pd.read_csv('Model/titanic_train.csv')
	return render_template("preview.html", df_view=df1.head(100))
	
@app.route('/css')
def css():
	return render_template('index.css')

	
@app.route('/predict', methods=['GET', 'POST'])
def predict():
	titanic = pd.read_csv('Model/titanic_train.csv')
	include = ['Age', 'Sex', 'Embarked', 'Survived']

	titanic_df = titanic[include]
	titanic_df.dropna(inplace = True)
	X = titanic_df.drop(['Survived'], axis = 1)
	y = titanic_df.Survived
	nom = LabelEncoder()
	X['Sex'] = nom.fit_transform(X['Sex'])
	X['Embarked'] = nom.fit_transform(X['Embarked'])
	logistic_model = open('Model/logit_model.pkl', 'rb')
	clf = joblib.load(logistic_model)
	
	if request.method == 'POST':
		Age = request.form['Age']
		Sex = request.form['Sex']
		Embarked = request.form['Embarked']
		Age = int(Age)
		Sex = int(Sex)
		Embarked = int(Embarked)
		vect = [Age, Sex, Embarked]
		vect=np.array(vect).reshape(1,-1)
		prediction = clf.predict(vect)
	return render_template("predict.html", prediction=prediction) 
	
if __name__ == "__main__":
	app.run(debug=True)