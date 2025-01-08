import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Load the custom model
model = load_model('custom_MobileNetV2_model.h5')

# List of the 90 animal classes
class_labels = [
    'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 
    'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 
    'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 
    'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 
    'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 
    'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 
    'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 
    'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 
    'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 
    'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 
    'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 
    'wolf', 'wombat', 'woodpecker', 'zebra'
]

# Function to preprocess the input image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))  # Match the input size used for training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to make a prediction
def predict_image(image_path):
    processed_img = preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(processed_img)

    # Get the predicted class (index) and corresponding label
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class_index]

    # Print the result
    print(f"Prediction: {predicted_label} ({predictions[0][predicted_class_index] * 100:.2f}% confidence)")

# Usage example
image_path = 'yourImage.jpg'
predict_image(image_path)
