from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os 

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for a home page


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('prediction.html')
    else:
        data = CustomData(
            year=int(request.form.get('yr', False)),
            team1=request.form.get('tm1', False),
            team2=request.form.get('tm2', False),
            city=request.form.get('city', False),
            toss=request.form.get('td', False),
            toss_win=request.form.get('tw', False)
        
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction check")
#   [[season, team1, team2, city, toss, toss_win, venue]]

        if pred_df['team1'][0] == pred_df['team2'][0]:
            msg = "Both Team1 and Team2 are same, Select different Teams and try again"
            print(msg)
            return render_template('prediction.html', prediction=msg)
            

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        pred_winner = predict_pipeline.predict(pred_df)
        print(f"Predicted winner is: {pred_winner}")
        return render_template('prediction.html', prediction=pred_winner)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
