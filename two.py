from flask import Flask, request, jsonify
import cv2
import ultralytics
import numpy as np
import os
import cv2
import numpy as np
import base64
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import tensorflow as tf


model_foot = ultralytics.YOLO("besti.pt")
model_acanthosis = ultralytics.YOLO("bestj.pt")

app = Flask(__name__)
CORS(app, support_credentials=True)

nail_model = tf.keras.models.load_model("nailmodelp.h5")
image_model = load_model('justtest_model.h5')
image_class_labels = ['Celliulitiescausedbydiabetes', 'Celliulitiescausednotbydiabetes']

def detect_objects(image, model):
    try:
        results = model(image)

        if results:
            json_response = {}
            for detection in results:
                try:
                    class_index = detection.boxes.cls[0].item()
                    if class_index in detection.names:
                        json_response['class'] = detection.names[class_index]
                    else:
                        json_response['class'] = 'Unknown Class'
               
                except IndexError:
                    json_response['error'] = 'Enter a valid Image'

            return json_response
    except Exception as e:
        return {'error': str(e)}


@app.route("/detect/foot", methods=["POST"])
def detect_foot():

    data = request.get_json()
    base64_image = data.get('image')
    image_data = base64.b64decode(base64_image)
    decoded_image = Image.open(BytesIO(image_data))
    image_file = decoded_image.convert("RGB")
    json_response = detect_objects(image_file, model_foot)
    return jsonify(json_response)

@app.route("/detect/acanthosis", methods=["POST"])
def detect_acanthosis_route():
    data = request.get_json()
    base64_image = data.get('image')

    image_data = base64.b64decode(base64_image)
    decoded_image = Image.open(BytesIO(image_data))
    image_file = decoded_image.convert("RGB")
    json_response = detect_objects(image_file, model_acanthosis)
    return jsonify(json_response)
    
@app.route('/upload1', methods=['POST'])
def upload_image():
    #try:
        data = request.get_json()
        base64_image = data.get('image')

        if base64_image:
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            image = image.resize((224, 224))
            image = image.convert("RGB")
            image_array = np.array(image)
            image_array = image_array / 255.0
            test_image = np.expand_dims(image_array, axis=0)
            predictions = image_model.predict(test_image)
            predicted_class_index = np.argmax(predictions, axis=1)
            predicted_class_label = image_class_labels[predicted_class_index[0]]
            confidence = np.amax(predictions)

            response = {
                "class_label": predicted_class_label,
            }
            return jsonify(response)
        else:
            return jsonify({'message': 'Invalid image data'})   

@app.route('/predict1', methods=['POST'])
def predict_nail():
     data = request.get_json()
     base64_image = data.get('image')

     if base64_image:
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))
        image = image.resize((224, 224))
        image = image.convert("RGB")
        predictions = nail_model.predict(np.expand_dims(image, axis=0))
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        class_names = ["Diabetic_Nail", "Non_Diabetic_Nail"] 
        predicted_label = class_names[predicted_class]

        response = {
            "predicted_label": predicted_label,
        }

        return jsonify(response)

     else:
        return jsonify({'message': 'Invalid image data'}) 

if __name__ == "__main__":
    app.run(debug=True)
