from fastapi import FastAPI, HTTPException
from utils import predict_price

app = FastAPI()

@app.post("/predict")
async def predict_price_endpoint(house_features: dict):
    try:
        # Call the predict_price function
        predicted_price = predict_price(house_features)
        return {"predicted_price": predicted_price}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
