#This part is to use training model to predict
import torch
import joblib
from models.train_fno_overunder import FNO1dOverUnder

scaler = joblib.load("data_preprocessing/scaler_overunder.pkl")

# 加载模型
def load_model(path="models/fno_overunder.pth"):
    model = FNO1dOverUnder(modes=8, width=32)
    try:
        model.load_state_dict(torch.load(path))
        model.eval()
    except Exception as e:
        print(f"Failed to load Over-Under model: {e}")
        raise
    return model

# input
def predict_over_under(odds_list, model=None):
    if model is None:
        model = load_model()
    odds_scaled = scaler.transform([odds_list])
    x = torch.tensor(odds_scaled, dtype=torch.float32).reshape(1, 1, 3)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()
        label = "Over" if prob > 0.5 else "Under"
    return label, prob

