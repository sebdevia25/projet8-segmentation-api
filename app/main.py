from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from .model import MODEL
from .utils import preprocess_image

app = FastAPI(title="Segmentation API")

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content = await image.read()
    X = preprocess_image(content)

    preds = MODEL.predict(X)
    mask = np.argmax(preds, axis=-1)[0]

    return JSONResponse({"prediction": mask.tolist()})
