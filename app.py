import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

# App Initialization
app = Flask(__name__)

# Load The Models
#with open('final_pipeline.pkl', 'rb') as file_1:
#  model_pipeline = joblib.load(file_1)

from tensorflow.keras.models import load_model
model_dnn_2 = load_model('model_dnn_2.tf')

# Route : Homepage
@app.route('/')
def home():
    return '<h1> It Works! </h1>'

@app.route('/predict', methods=['POST'])
def dnn_predict():
    args = request.json
    print(args, dir(args))

    X_inf = { 
        'id': args.get('id'),
        'keyword': args.get('keyword'),
        'location': args.get('location'),
        'text': args.get('text')
    }


    print('[DEBUG] Data Inference : ', X_inf)

    
    # Transform Inference-Set
    X_inf = pd.DataFrame([X_inf])
    X_inf.drop('id', axis=1, inplace=True)
    X_inf.drop('keyword', axis=1, inplace=True)
    X_inf.drop('location', axis=1, inplace=True)
    y_pred_model_dnn_2 = model_dnn_2.predict(X_inf)
    y_pred_model_dnn_2 = np.where(y_pred_model_dnn_2 >=0.5, 1, 0)


    if y_pred_model_dnn_2 == 0:
        label = 'Not disaster tweet'
    else:
        label ='Disaster tweet'

    print('[DEBUG] Result : ', y_pred_model_dnn_2, label)
    print('')

    response = jsonify(
        result = str(y_pred_model_dnn_2),
        label_names = label
    )

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0')