from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
import numpy as np
import tensorflow as tf
import pickle

app = FastAPI()


with open("Prediction_model.pkl", "rb") as f:
    model = pickle.load(f)


# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load your model
# model = tf.keras.models.load_model("Prediction_model.pkl")  # Replace with actual model filename

# Class names (replace with your classes)
class_names = ['Early Blight','Late Blight', 'Healthy']

def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())

    # Resize and normalize the image as per your model requirements
    image = image.resize((128, 128))  # Use model's expected input size
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {"class": predicted_class, "confidence": confidence}
