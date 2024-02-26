from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# For potato
MODEL = tf.keras.models.load_model("../Potato/models/potato.keras")
print(MODEL)

# For potato
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/test")
async def test():
    return "API is running..."


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    print(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    accuracy = np.max(predictions[0])
    print(accuracy)
    return {"class": predicted_class, "accuracy": float(accuracy)}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
