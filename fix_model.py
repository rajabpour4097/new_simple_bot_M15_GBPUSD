"""
تبدیل مدل به نسخه جدید sklearn
"""
import joblib
import pickle
from pathlib import Path

# مسیر مدل
model_path = Path('ml_models/best_model.pkl')

if model_path.exists():
    try:
        # تلاش برای بارگذاری
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Type: {type(model)}")
        
        # ذخیره مجدد با نسخه جدید
        joblib.dump(model, model_path)
        print(f"✅ Model re-saved with new sklearn version")
        
    except Exception as e:
        print(f"❌ Cannot load model: {e}")
        print("\n💡 Solution: Retrain the model with current sklearn version")
        print("   Run: python ../ml_train_models.py")
else:
    print(f"❌ Model not found at {model_path}")
