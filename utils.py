import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

# Load all models

nationality_model = load_model(
    "models/nationality_model.keras"
)

emotion_model = load_model(
    "models/emotion_model.keras"
)

age_model = load_model(
    "models/age_model.keras"
)

dress_color_model = load_model(
    "models/dress_color_model.keras"
)

print("✅ All models loaded successfully!")


def preprocess_image(image, target_size):
    """
    Resize and normalize image for model prediction.
    """

    # Resize
    image = cv2.resize(image, target_size)

    # Normalize
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


def preprocess_emotion_image(image):
    """
    Convert image to grayscale and preprocess for emotion model.
    """

    # Convert RGB to Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize to 48x48
    image = cv2.resize(image, (48, 48))

    # Normalize
    image = image.astype("float32") / 255.0

    # Add channel dimension
    image = np.expand_dims(image, axis=-1)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

def crop_clothing_region(image):
    """
    Crop the approximate upper-body clothing region.
    """

    h, w = image.shape[:2]

    # Approximate upper-body crop
    top = int(h * 0.25)
    bottom = int(h * 0.75)

    left = int(w * 0.20)
    right = int(w * 0.80)

    clothing = image[top:bottom, left:right]

    return clothing

def predict_nationality(image):
    """
    Predict nationality from an RGB image.
    """

    # Class names in the same order used during training
    class_names = ['African', 'Indian', 'Others', 'United States']

    # Preprocess image
    input_image = preprocess_image(image, (224, 224))

    # Predict
    prediction = nationality_model.predict(input_image, verbose=0)

    # Get predicted class index
    predicted_index = np.argmax(prediction)

    # Get predicted class name
    predicted_class = class_names[predicted_index]

    # Get confidence score
    confidence = float(np.max(prediction))

    return predicted_class, confidence


def predict_emotion(image):
    """
    Predict emotion from an RGB image.
    """

    emotion_labels = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral"
    }

    # Preprocess image
    input_image = preprocess_emotion_image(image)

    # Predict
    prediction = emotion_model.predict(input_image, verbose=0)

    # Highest probability index
    predicted_index = np.argmax(prediction)

    # Emotion name
    predicted_emotion = emotion_labels[predicted_index]

    # Confidence
    confidence = float(np.max(prediction))

    return predicted_emotion, confidence


def predict_age(image):
    """
    Predict age group from an RGB image.
    """

    age_labels = {
        0: "0-5",
        1: "11-15",
        2: "16-20",
        3: "21-30",
        4: "31-40",
        5: "41-50",
        6: "51-60",
        7: "6-10",
        8: "61-70",
        9: "71-80",
        10: "81+"
    }

    # Preprocess image
    input_image = preprocess_image(image, (160, 160))

    # Predict
    prediction = age_model.predict(input_image, verbose=0)

    predicted_index = np.argmax(prediction)

    predicted_age = age_labels[predicted_index]

    confidence = float(np.max(prediction))

    return predicted_age, confidence

# predict_dress_color()
def predict_dress_color(image):
    """
    Predict dress color from an RGB image.
    """

    color_labels = {
        0: "Black",
        1: "Blue",
        2: "Brown",
        3: "Green",
        4: "Grey",
        5: "Pink",
        6: "Purple",
        7: "Red",
        8: "White",
        9: "Yellow"
    }

    # Crop clothing region
    clothing = crop_clothing_region(image)

    # Preprocess
    input_image = preprocess_image(clothing, (128, 128))

    # Predict
    prediction = dress_color_model.predict(input_image, verbose=0)

    predicted_index = np.argmax(prediction)

    predicted_color = color_labels[predicted_index]

    confidence = float(np.max(prediction))

    return predicted_color, confidence

# Add predict()
def predict(image):
    """
    Complete prediction pipeline.
    """

    results = {}

    # Step 1: Nationality
    nationality, nat_conf = predict_nationality(image)

    results["Nationality"] = nationality
    results["Nationality Confidence"] = round(nat_conf * 100, 2)

    # Step 2: Emotion (for everyone)
    emotion, emo_conf = predict_emotion(image)

    results["Emotion"] = emotion
    results["Emotion Confidence"] = round(emo_conf * 100, 2)

    # Step 3: Nationality-specific predictions

    if nationality == "Indian":

        age, age_conf = predict_age(image)
        color, color_conf = predict_dress_color(image)

        results["Age"] = age
        results["Age Confidence"] = round(age_conf * 100, 2)

        results["Dress Color"] = color
        results["Dress Color Confidence"] = round(color_conf * 100, 2)

    elif nationality == "United States":

        age, age_conf = predict_age(image)

        results["Age"] = age
        results["Age Confidence"] = round(age_conf * 100, 2)

    elif nationality == "African":

        color, color_conf = predict_dress_color(image)

        results["Dress Color"] = color
        results["Dress Color Confidence"] = round(color_conf * 100, 2)

    return results


# Test Nationality + Emotion Together
if __name__ == "__main__":

    IMAGE_PATH = r"FairFace_Dataset/FairFace/train/10003.jpg"   # Change to your test image

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("❌ Image not found!")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = predict(image)

        print("\n========== Prediction ==========\n")

        for key, value in results.items():
            print(f"{key}: {value}")