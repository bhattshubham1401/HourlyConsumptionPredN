import traceback
from datetime import datetime

from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # initializing a flask app


@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:

            #  reading the inputs given by the user
            Clock = datetime.strptime(request.form['Clock'], '%Y-%m-%d')
            sensor = str(request.form['sensor'])
            R_Voltage = float(request.form['R_Voltage'])
            Y_Voltage = float(request.form['Y_Voltage'])
            B_Voltage = float(request.form['B_Voltage'])
            R_Current = float(request.form['R_Current'])
            Y_Current = float(request.form['Y_Current'])
            B_Current = float(request.form['B_Current'])

            data = [Clock, sensor, R_Voltage, Y_Voltage, B_Voltage, R_Current, Y_Current, B_Current]
            data = np.array(data).reshape(1, 11)

            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction=str(predict))

        except Exception as e:
            print(traceback.format_exc())
            print('The Exception message is: ', e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port = 8080, debug=True)
    app.run(host="0.0.0.0", port=8080)
