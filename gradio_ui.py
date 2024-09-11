import gradio as gr
import requests
import cv2
import numpy as np

# Define Gradio interface components
input_image = gr.components.Image(label="Upload Image")
output_image = gr.components.Image(label="Output Image")

# Define the function to call the FastAPI backend
def detect_objects(image):
        
    image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
    image_file = {"file": ("image.jpg", image_bytes, "image/jpeg")}

    response = requests.post("http://localhost:8000/detect/", files=image_file)
    
    image_array = np.frombuffer(response.content, dtype=np.uint8)
    
    decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return decoded_image

path=[['image-1.jpg'],['image-2.jpg']]

desc = "This Gradio app takes ariel satellite images as input and checks for wildfire and cyclone. " 


# Create a Gradio interface
ui=gr.Interface(
    fn=detect_objects,
    inputs=input_image,
    outputs=output_image,
    allow_flagging='never',
    description= desc,
    examples=path,
    cache_examples=False,
    title = "Cyclone and Wildfire Detector",
)
