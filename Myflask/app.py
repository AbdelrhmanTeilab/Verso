"""from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Initialize Flask application
app = Flask(__name__)

classes = {
  'Zinc deficiency': r'D:/Dataset/new/Beau_ag',
    'Lung abscess and bacteria in the mouth': r'D:/Dataset/new/clubbing_ag',
    'Exposure to toxins': r'D:/Dataset/new/mees_line_ag',
    'normal': r'D:/Dataset/new/normal_ag',
    'Skin inflammation': r'D:/Dataset/new/onycholysis_ag',
    'Kidney or liver problems consult a doctor': r'D:/Dataset/new/terrys_nail_ag',
    'In critical condition consult a doctor immediately': r'D:/Dataset/new/black_line_ag',
    'Calcium deficiency': r'D:/Dataset/new/white_spot_ag',
    # Add more classes as needed
}
# Load the pre-trained model
model_path = 'C:/Users/abdel/Downloads/Myflask/Verso_model.h5'
loaded_model = None
class_names = list(classes.keys())  # Get the class names from global variable 'classes'

def load_my_model():
    global loaded_model
    loaded_model = load_model('C:/Users/abdel/Downloads/Myflask/Verso_model.h5')

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Route to predict the image class
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided'})

        # Save the file to a temporary location
        img_path = 'D:/Games/aug_0_104.jpg'
        file.save(img_path)

        # Preprocess the image
        img_array = preprocess_image(img_path)

        # Predict the class probabilities
        predictions = loaded_model.predict(img_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Get the predicted class name and probability
        predicted_class_name = class_names[predicted_class_index]
        predicted_probability = float(predictions[0][predicted_class_index])

        # Return the result as JSON
        return jsonify({
            'class_name': predicted_class_name,
            'probability': predicted_probability
        })

if __name__ == '__main__':
    load_my_model()  # Load the model before starting the Flask app
    app.run(debug=True)
"""


'''from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your AI model (fine-tuned model)
model_path = 'C:/Users/abdel/Downloads/Myflask/Verso_model.h5'
loaded_model = tf.keras.models.load_model('C:/Users/abdel/Downloads/Myflask/Verso_model.h5')

# Define your classes
classes = {
    0: 'Zinc deficiency',
    1: 'Lung abscess and bacteria in the mouth',
    2: 'Exposure to toxins',
    3: 'normal',
    4: 'Skin inflammation',
    5: 'Kidney or liver problems consult a doctor',
    6: 'In critical condition consult a doctor immediately',
    7: 'Calcium deficiency',
    # Add more classes as needed
}

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale to [0, 1]
    return img_array

# Route to predict the image class
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            # Save the file to a temporary location
            img_path = 'D:/Games/aug_0_104.jpg'
            file.save(img_path)

            # Preprocess the image
            img_array = preprocess_image(img_path)

            # Predict the class probabilities
            predictions = loaded_model.predict(img_array)

            # Get the predicted class index
            predicted_class_index = np.argmax(predictions)

            # Get the predicted class name and probability
            predicted_class_name = classes[predicted_class_index]
            predicted_probability = float(predictions[0][predicted_class_index])

            # Return the result as JSON
            return jsonify({
                'class_name': predicted_class_name,
                'probability': predicted_probability
            })

if __name__ == '__main__':
    app.run(debug=True)'''


'''import base64
from flask import Flask,request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

new_model=tf.keras.models.load_model("C:/Users/abdel/Downloads/Myflask/Verso_model.h5")

app= Flask(__name__)


@app.route('/api',methods=['put'])

def index ():
    inputchar= request.get_data()
    imgdata= base64.b64decode(inputchar)
    filename="imag.jpg"
    with open(filename,'wb') as f:
        f.write(imgdata)

    return 'hallo'
if __name__ == '__main__':
    app.run(debug=True)'''

from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image  # Import Image from PIL

# Initialize Flask application
app = Flask(__name__)

# Load the saved fine-tuned model
loaded_model = tf.keras.models.load_model('C:/Users/abdel/Downloads/Myflask/Verso_model.h5')

# Define the classes
classes = {
    0: 'Zinc deficiency',
    1: 'Lung abscess and bacteria in the mouth',
    2: 'Exposure to toxins',
    3: 'normal',
    4: 'Skin inflammation',
    5: 'Kidney or liver problems consult a doctor',
    6: 'In critical condition consult a doctor immediately',
    7: 'Calcium deficiency',
    # Add more classes as needed
}

# Define a function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)  # Use Image from PIL to open the image
    img = img.resize((224, 224))  # Resize the image to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale to [0, 1]
    return img_array

# Define a function to classify a new image
def classify_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = loaded_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = classes[predicted_class_index]
    predicted_probability = predictions[0][predicted_class_index]
    return predicted_class_name, float(predicted_probability)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is received
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Check if the file is not empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save the uploaded file to a temporary location
        temp_image_path = 'D:/Games/aug_0_104.jpg'
        file.save(temp_image_path)

        # Classify the uploaded image
        predicted_class, predicted_prob = classify_image(temp_image_path)

        # Return the result as JSON response
        return jsonify({'class': predicted_class, 'probability': predicted_prob}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
