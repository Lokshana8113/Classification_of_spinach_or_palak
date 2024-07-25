from flask import Flask, render_template, request
from google.colab.output import eval_js
from base64 import b64decode
from keras.models import load_model
import cv2
import numpy as np
from pyngrok import ngrok

# Set your ngrok authentication token
!ngrok authtoken YOUR_AUTHTOKEN

app = Flask(__name__)

# Load the pre-trained model
model = load_model('models/palak_model_100epochs.h5')

# Define image preprocessing function
def preprocess_image(img):
    # Resize the image to a fixed size (e.g., 150x150)
    img_resized = cv2.resize(img, (150, 150))
    # Normalize pixel values to be between 0 and 1
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data
    data_url = request.form['image']
    header, encoded = data_url.split(",", 1)
    data = b64decode(encoded)

    # Convert image data to OpenCV format
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess the image
    img_preprocessed = preprocess_image(img)

    # Make prediction
    prediction = model.predict(np.expand_dims(img_preprocessed, axis=0))
    prediction_label = "Dry" if prediction > 0.5 else "Fresh"

    # Return prediction result
    return "Prediction: {}".format(prediction_label)

# Start ngrok tunnel
public_url = ngrok.connect(5000)

# Run Flask app
app.run()
