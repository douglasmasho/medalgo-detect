# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 17:46:57 2024

@author: Douglas Masho
"""

import asyncio
import logging
import signal
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import numpy as np
from ultralytics import YOLOv10
from PIL import ImageOps


# Set event loop policy for Windows
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_api")

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLOv10 model
model_path = "yolodetection.pt"
logger.info(f"Loading YOLOv10 model from {model_path}")
model = YOLOv10(model_path)

# Define the prediction function
def predict(image):
    try:
 logger.info("Converting uploaded image to RGB format")
        image = Image.open(io.BytesIO(image)).convert("RGB")

        # Resize the image to 640x640 before converting to a numpy array
        logger.info("Resizing the image to 640x640")
        image = ImageOps.fit(image, (640, 640), Image.ANTIALIAS)
        
        image = np.array(image)

        logger.info("Running YOLOv10 model prediction")
        result = model.predict(source=image, imgsz=640, conf=0.25)

        logger.info("Annotating the image with YOLOv10 predictions")
        annotated_img = result[0].plot()
        annotated_img = Image.fromarray(annotated_img[:, :, ::-1])

        img_byte_arr = io.BytesIO()
        annotated_img.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise


@app.get("/")
async def health_check():
    return "The health check is successful!"


@app.get("/health")
async def health_check():
    return "The health check is successful!"

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        image_bytes = await file.read()
        logger.info("Making prediction on the uploaded image")
        annotated_image = predict(image_bytes)
        logger.info("Returning annotated image as response")
        return StreamingResponse(io.BytesIO(annotated_image), media_type="image/jpeg")
    except ConnectionResetError as e:
        logger.error(f"Connection reset error: {str(e)}")
        return {"error": "Connection reset error. Please try again."}
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {str(e)}")
        return {"error": str(e)}

def shutdown(signum, frame):
    logger.info(f"Received signal {signum}. Shutting down gracefully...")
    uvicorn.Server.should_exit = True

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=120)
