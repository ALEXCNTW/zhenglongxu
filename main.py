
from utils.odds_scraper import get_real_odds
from models.predict import predict_result
from models.predict_overunder import predict_over_under

odds = [1.90, 3.30, 4.20]
label, confidence = predict_over_under(odds)

print("Prediction result：", label)
print("Probability of predicting Over：", confidence)



# Win-Draw-Loss prediction
label_result, prob_result = predict_result(odds)
print(f"🏆 Win-Draw-Loss prediction：{label_result}（Confidence level：{prob_result:.2f}）")

# Over-Under prediction
label_ou, prob_ou = predict_over_under(odds)
print(f"Over-Under prediction：{label_ou}（Confidence level：{prob_ou:.2f}）")
