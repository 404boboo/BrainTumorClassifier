import tensorflow as tf
import numpy as np
import os
from PIL import Image
import argparse

MODEL_PATH = "finalModel.keras"

def preprocess_image(image_path, img_height=180, img_width=180):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")
    
    img = img.resize((img_width, img_height))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array = img_array / 255.0  # Rescaling as per training
    return img_array

def predict_image(model, img_array, class_names):
    """
    Predicts the class of the image using the trained model.
    """
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]
    return predicted_class, confidence

def main():
    parser = argparse.ArgumentParser(description='Classify a Brain Tumor using a trained TensorFlow model.')
    parser.add_argument('--image', type=str, required=True, help='Path to the image to classify.')
    args = parser.parse_args()

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']  

    img_array = preprocess_image(args.image)

    predicted_class, confidence = predict_image(model, img_array, class_names)

    print(f"Predicted Class: {predicted_class} with confidence {confidence*100:.2f}%")

if __name__ == "__main__":
    main()