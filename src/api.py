"""
AFL Betting Model API - serves over/under probabilities for player disposals.
Endpoints:
  POST /predict - get probabilities for a player-game
  GET  /health - health check
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="AFL Betting Model", version="0.1")

# Load models and features on startup
MODELS_DIR = "models/context"
models = {}
feature_cols = None

def load_models():
    global models, feature_cols
    model_path = os.path.join(MODELS_DIR, 'models_context.pkl')
    feat_path = os.path.join(MODELS_DIR, 'features.pkl')
    if os.path.exists(model_path) and os.path.exists(feat_path):
        models = joblib.load(model_path)
        feature_cols = joblib.load(feat_path)
        print(f"Loaded models for targets: {list(models.keys())}")
        print(f"Feature columns: {feature_cols}")
    else:
        print("Warning: models not found in models/context. Please run backtest to generate.")

load_models()

class PredictRequest(BaseModel):
    year: int
    round: int
    player: str
    team: str
    opponent: str
    venue: str
    is_home: int = 0
    # Context features (can be computed client-side or via lookup)
    had_bye_last_week: int = 0
    opponent_avg_disposals: float = None
    opponent_defensive_avg: float = None
    venue_avg_disposals: float = None
    # Rolling features (can be provided, or we'll compute from historical if available)
    disposals_last_5: float = 0.0
    disposals_std_5: float = 0.0
    disposals_last_10: float = 0.0
    disposals_std_10: float = 0.0
    disposals_last_20: float = 0.0
    disposals_std_20: float = 0.0
    disposals_prev: float = 0.0

class PredictResponse(BaseModel):
    over_195_prob: float
    over_245_prob: float
    over_295_prob: float
    over_195_pred: int
    over_245_pred: int
    over_295_pred: int

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(models) > 0}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not models or feature_cols is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    # Build feature vector
    data = req.dict()
    # Ensure all feature cols present
    try:
        X = pd.DataFrame([{col: data.get(col, 0) for col in feature_cols}])
        X = X.replace([float('inf'), float('-inf')], pd.NA).fillna(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature construction error: {e}")
    preds = {}
    for target in ['over_195','over_245','over_295']:
        model = models[target]
        prob = float(model.predict_proba(X)[0,1])
        pred = int(model.predict(X)[0])
        preds[f'{target}_prob'] = prob
        preds[f'{target}_pred'] = pred
    return PredictResponse(**preds)

# For local running
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
