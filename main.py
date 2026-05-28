# imports

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import numpy as np
import cv2

import base64
from io import BytesIO
from PIL import Image

# app

app = FastAPI()


# cors

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def get_landslide_score(pred_mask):
    return float(np.mean(pred_mask))


def predict_risk(score, iot):

    if score > 0.6 and iot["moisture"] > 70:
        return "HIGH"

    elif score > 0.4 or iot["rainfall"] > 50:
        return "MEDIUM"

    else:
        return "LOW"

model = tf.keras.models.load_model(
    "landslide_unet.h5"
)

# ROUTE

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    npimg = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (128,128))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    probability = float(np.mean(pred))

    # Create binary mask
    mask = (pred[0].squeeze() > 0.5).astype(np.uint8) * 255

    # Convert mask to image
    mask_image = Image.fromarray(mask)

    # Convert image to base64
    buffer = BytesIO()

    mask_image.save(buffer, format="PNG")

    mask_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return {
        "probability": probability,
        "mask": mask_base64
    }