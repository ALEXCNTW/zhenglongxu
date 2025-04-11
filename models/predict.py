#This part is to use training model to predict
import torch
import joblib
from models.train_fno import FNO1dClassifier as FNO1d

# loading scaler
scaler = joblib.load("data_preprocessing/scaler.pkl")

def load_model(path="models/fno_model.pth"):
    model = FNO1d(modes=1, width=32)
    try:
        model.load_state_dict(torch.load(path))
        model.eval()
    except Exception as e:
        print(f"Failed to load Win-Draw-Loss model: {e}")
        raise
    return model

#Prediction
def predict_result(odds_list, model=None):
    if model is None:
        model = load_model()
    odds_scaled = scaler.transform([odds_list])
    x = torch.tensor(odds_scaled, dtype=torch.float32).reshape(1, 1, 3)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy().flatten()
        idx = probs.argmax()
    label_map = {0: "Home win", 1: "Draw", 2: "Away win"}
    return label_map[idx], probs[idx]

