import flask
from flask import Flask, jsonify, request
import json
from data_input import data_in
import numpy as np
import pickle
from waitress import serve



def load_models():
    file_name = "../Resources/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model

app = Flask(__name__)
@app.route('/predict', methods=['GET'])
def predict():
    # stub input features
    request_json = request.get_json()
    x = request_json['input']
    x[22] = np.log(x[22])
    #print(x)
    x_in = np.array(x).reshape(1,-1)
    # load model
    model = load_models()
    prediction = np.exp(model.predict(x_in)[0])
    response = json.dumps({'response': f'R$ {prediction:.2f}'})
    return response, 200

if __name__ == '__main__':
    #application.run(debug=True)
    serve(app,host='0.0.0.0', port=8080, threads=1)