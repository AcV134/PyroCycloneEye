import io
from fastapi import FastAPI
import cv2
from fastapi.responses import StreamingResponse
import numpy as np
from ultralytics import YOLO
import gradio as gr
from gradio_ui import ui
from twilio.rest import Client
app = FastAPI()

from fastapi import UploadFile

# Load the YOLO models for cyclone and wildfire detection
def Model_1(image1):
  model_01 = YOLO("Detection_best_1.pt")
  detect_01 = model_01.predict(image1)
  return detect_01

def Model_2(image2):
   model_02 = YOLO("Detection_best_2.pt")
   detect_02 = model_02.predict(image2)
   return detect_02
  
  
@app.post("/detect/")
async def detect_objects(file: UploadFile):
  # Process the uploaded image for object detection
  image_bytes = await file.read()
  image = np.frombuffer(image_bytes, dtype=np.uint8)
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  
  # Detect objects using Model-1 and Model-2
  detections_model_1 = Model_1(image)
  detections_model_2 = Model_2(image)
  
  
  # Get the maximum confidence scores
  max_conf_model_1 = max([box.conf.item() for r in detections_model_1 for box in r.boxes], default=0)
  max_conf_model_2 = max([box.conf.item() for r in detections_model_2 for box in r.boxes], default=0)
  
  model_01 = YOLO("Detection_best_1.pt")
  model_02 = YOLO("Detection_best_2.pt")
  
   # Choose the model with the higher confidence score
  if max_conf_model_1 >= max_conf_model_2:
    chosen_model = model_01
    chosen_detections = detections_model_1
  else:
    chosen_model = model_02
    chosen_detections = detections_model_2
      
  print(max_conf_model_1, max_conf_model_2)     
  conf_score=max(max_conf_model_1, max_conf_model_2)
  conf_str= str(conf_score)
  
  # ploting the boxes
  for r in chosen_detections:
    boxes = r.boxes
    labels = []
    for box in boxes:
      c = box.cls
      l = chosen_model.names[int(c)]
      labels.append(l)
  frame = chosen_detections[0].plot()
  
  
  # #Alert System
  # if conf_score>0.5:
    
  #   account_sid = ''
  #   auth_token = ''
  #   client = Client(account_sid, auth_token)
    
  #   from_whatsapp_no ='whatsapp:+14155238886'
  #   to_whatsapp_no ='whatsapp:+919567095966'
    
  #   message=client.messages.create(body=l+' has been detected in your area with a probaility of '+conf_str+' please contact the weather department for confirmation.',
  #                          from_=from_whatsapp_no,
  #                          to=to_whatsapp_no)
  #   print(message)

  # Convert the original image array to bytes
  _, img_encoded = cv2.imencode('.jpg', frame)
  img_bytes = img_encoded.tobytes()
  if img_bytes== 0:
      return 

  # Send the original image back as a streaming response
  return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")

app=gr.mount_gradio_app(app,ui,path='')
