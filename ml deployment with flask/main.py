import sys
import joblib
import traceback
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/predict", methods = ['POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value = 0)

            prediction = list(lr.predict(query))
            return jsonify({'prediction': str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exec()})
    else:
        print("No model to use here.")


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) #Command line input
    except:
        port = 12345

    lr = joblib.load('models/model_columns.pkl')
    
    
    app.run(debug = True, port = port)