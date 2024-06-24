import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
titanic_model=pickle.load(open('titanic_model.pkl','rb'))
titanic_scaling=pickle.load(open('titanic_scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    # Convert data to a numpy array and reshape it for the model
    new_data = np.array(list(data.values())).reshape(1, -1)
    new_data_scaled = titanic_scaling.transform(new_data)
    # Make prediction
    output = titanic_model.predict(new_data_scaled)
    print(output[0])
    # Convert the output to a native Python type
    output_python = output[0].item()  # This will convert numpy data types to native Python data types
    return jsonify(output_python)

'''def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1) )
    new_data=titanic_scaling.transform(np.array(list(data.values())).reshape(1, -1) )
    output=titanic_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])'''

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=titanic_scaling.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=titanic_model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Output is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)