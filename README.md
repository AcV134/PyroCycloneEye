# PyroCycloneEye

- An early detection system for wildfire and cyclone using deeplearning.
- Dataset used is satellite images of cloud patterns and smoke(from fire).
- ML model used is YOLOV8 which focuses on object detection, image segmentation etc.
- Frontend is developed using gradio.
- Backend server is built using fastapi.

## Installation
- Fork this repositary
- start a virtual machine:
```Python
python -m venv (vname)
.\(vname)\Scripts\activate
```
- run `requirements.txt`
```Python
pip install -r requirements.txt
```
-To start run app.py using:
```Python
uvicorn app:app --reload
```
