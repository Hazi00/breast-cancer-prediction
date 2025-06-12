from flask import Flask, request, render_template
import pickle
import numpy as np


app = Flask(__name__)

model = pickle.load(open('breast_cancer\model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')





@app.route('/predict', methods=['GET', 'POST'])
def predict():
        
    mean_radius = request.form['mean_radius']
    mean_texture = request.form['mean_texture']
    mean_perimeter = request.form['mean_perimeter']
    mean_area = request.form['mean_area']
    mean_smoothness = request.form['mean_smoothness']
    arr = np.array([mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])
    
    if pred == 1:
        result = "Yes"
    else:
        result = "No"
    return render_template('index.html', prediction=result)


if __name__ == '_main_':
    app.run(debug=True)