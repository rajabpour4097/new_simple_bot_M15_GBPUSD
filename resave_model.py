"""
ذخیره مجدد مدل با sklearn جدید
"""
import joblib
from pathlib import Path

model_path = Path('ml_models/best_model.pkl')
scaler_path = Path('ml_models/scaler.pkl')

# بارگذاری
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print(f"✅ Model loaded: {type(model)}")
print(f"✅ Scaler loaded: {type(scaler)}")

# ذخیره مجدد با نسخه جدید
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n✅ Re-saved with sklearn 1.8.0")
print(f"   {model_path}")
print(f"   {scaler_path}")
