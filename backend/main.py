from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Is 'app' variable ko Uvicorn dhoond raha hai
app = FastAPI(title="Chest X-Ray Pneumonia Detector")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model = tf.keras.models.load_model("chest_xray_model.keras")

@app.get("/")
def home():
    return {"message": "Chest X-Ray Backend is Running!"}

@app.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize((150, 150))
        
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        if prediction > 0.5:
            conf = float(prediction * 100)
            return {
                "success": True,
                "result": "PNEUMONIA DETECTED",
                "confidence": f"{conf:.2f}%",
                "is_pneumonia": True
            }
        else:
            conf = float((1 - prediction) * 100)
            return {
                "success": True,
                "result": "NORMAL X-RAY",
                "confidence": f"{conf:.2f}%",
                "is_pneumonia": False
            }
            
    except Exception as e:
        return {"success": False, "error": str(e)}