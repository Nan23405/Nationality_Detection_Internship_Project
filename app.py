import gradio as gr
import cv2
import numpy as np

from utils import predict

def analyze_image(image):

    if image is None:
        return "Please upload an image."

    results = predict(image)

    output = ""

    for key, value in results.items():
        output += f"{key}: {value}\n"

    return output

interface = gr.Interface(
    fn=analyze_image,

    inputs=gr.Image(type="numpy", label="Upload Image"),

    outputs=gr.Textbox(label="Prediction Results"),

    title="🌍 AI-Based Nationality Detection System",

    description="""
Upload a face image to predict:

• Nationality

• Emotion

• Age (for Indian & United States)

• Dress Color (for Indian & African)
"""
)

if __name__ == "__main__":
    interface.launch()