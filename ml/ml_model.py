from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)  

# Load trained model
model = tf.keras.models.load_model('pneumonia_resnet50.h5')
print("Model loaded successfully!")

# Image preprocessing
def prepare_image(image_file):
    img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file'}), 400

    try:
        img = prepare_image(file)
        prediction = model.predict(img)[0][0]

        result = 'PNEUMONIA' if prediction >= 0.5 else 'NORMAL'
        confidence = float(prediction if result=='PNEUMONIA' else 1-prediction)

        return jsonify({
            'prediction': result,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
