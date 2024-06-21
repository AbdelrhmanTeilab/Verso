
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the classes and their corresponding paths
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

# Image dimensions and batch size
img_height, img_width = 224, 224
batch_size = 32

# Create a data generator with augmentation for fine-tuning
fine_tune_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% training, 20% validation split
)

# Load and augment training data for fine-tuning
fine_tune_train_generator = fine_tune_datagen.flow_from_directory(
    'D:/Dataset/new',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Load and augment validation data for fine-tuning
fine_tune_val_generator = fine_tune_datagen.flow_from_directory(
    'D:/Dataset/new',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load the pre-trained InceptionV3 model without top layers
base_model = tf.keras.applications.InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

# Freeze the first layers of the pre-trained model
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Add new layers for symptoms and diseases classification
x = layers.Flatten()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
num_classes = len(classes)  # Number of classes based on provided classes
output_layer = layers.Dense(num_classes, activation='softmax')(x)

# Create the final model for fine-tuning
fine_tuned_model = Model(base_model.input, output_layer)

# Compile the model for fine-tuning
fine_tuned_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

# Display the summary of the final model for fine-tuning
fine_tuned_model.summary()
# Fine-tune the model
fine_tuned_history = fine_tuned_model.fit(
    fine_tune_train_generator,
    epochs=20,  # You can adjust the number of epochs based on training performance
    validation_data=fine_tune_val_generator
)

# Save the model in the native Keras format
fine_tuned_model.save('C:/Users/abdel/Downloads/New folder/fine_tuned_model.tflite')

# Load the model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved fine-tuned model
loaded_model = tf.keras.models.load_model('C:/Users/abdel/Downloads/New folder/fine_tuned_model.h5')

# Define a function to classify a new image
def classify_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale to [0, 1]

    # Predict the class probabilities
    predictions = loaded_model.predict(img_array)
    
    # Get the class names
    class_names = list(classes.keys())

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]

    # Get the probability of the predicted class
    predicted_probability = predictions[0][predicted_class_index]

    return predicted_class_name, predicted_probability

# Example usage
image_path = 'D:/Games/aug_0_104.jpg'  # Replace with the path to your image
result = classify_image(image_path)
print("Predicted Class:", result[0])
print("Probability:", result[1])


