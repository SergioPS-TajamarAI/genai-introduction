from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load the pre-trained Keras model
model = tf.keras.models.load_model("./image_model.keras")


@app.get("/image_classification", response_class=HTMLResponse)
async def get_image_classification_form():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Classification</title>
    </head>
    <body>
        <h1>Draw an Image (64x64)</h1>
        <form action="/image_classification" method="post" enctype="multipart/form-data">
            <label for="file">Upload your image:</label>
            <input type="file" id="file" name="file" accept="image/*" required>
            <button type="submit">Classify</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/image_classification")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Leer el archivo subido
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("L")  # Convertir a escala de grises
        image = image.resize((28, 28))  # Redimensionar a 28x28
        image_array = np.array(image) / 255.0  # Normalizar valores de píxeles
        image_array = np.expand_dims(image_array, axis=(0, -1))  # Agregar dimensiones de batch y canal

        # Predecir usando el modelo
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return JSONResponse(content={"PREDICTED NUMBER:": int(predicted_class)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
