"""
ماژول مدیریت پوزیشن با ML
فقط برای معاملات Reversed اعمال می‌شود
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import MetaTrader5 as mt5
from datetime import datetime, timedelta


class MLPositionManager:
    """مدیریت هوشمند SL/TP با استفاده از Machine Learning"""
    
    def __init__(self, symbol='GBPUSD'):
        """
        راه‌اندازی مدیر ML
        
        Args:
            symbol: نماد معاملاتی (GBPUSD یا EURUSD)
        """
        self.symbol = symbol
        self.model = None
        self.scaler = None
        self.enabled = False
        
        # بارگذاری مدل
        self._load_model()
    
    def _load_model(self):
        """بارگذاری مدل ML از دیسک"""
        try:
            # مسیر مدل‌ها - در همان پوشه ربات
            current_dir = Path(__file__).parent
            model_path = current_dir / 'ml_models' / 'best_model.pkl'
            scaler_path = current_dir / 'ml_models' / 'scaler.pkl'
            
            if not model_path.exists():
                print(f"⚠️ ML Model not found at {model_path}")
                print(f"   Searched in: {current_dir / 'ml_models'}")
                print(f"   ML position management DISABLED")
                return
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            self.enabled = True
            
            print(f"✅ ML Model loaded: {model_path.name}")
            print(f"✅ Scaler loaded: {scaler_path.name if self.scaler else 'None'}")
            print(f"🤖 ML Position Management: ENABLED for {self.symbol}")
            
        except Exception as e:
            print(f"❌ Failed to load ML model: {e}")
            print(f"   ML position management DISABLED")
    
    def should_apply_ml(self, is_reversed):
        """
        آیا باید ML اعمال شود؟
        
        Args:
            is_reversed: آیا معامله Reversed است؟
        
        Returns:
            bool: True اگر باید ML اعمال شود
        """
        # فقط برای Reversed و اگر مدل بارگذاری شده
        return self.enabled and is_reversed
    
    def adjust_sl_tp(self, entry, sl, tp, trade_type, is_reversed=False):
        """
        تنظیم SL و TP با استفاده از ML
        
        Args:
            entry: قیمت ورود
            sl: Stop Loss اولیه
            tp: Take Profit اولیه
            trade_type: نوع معامله ('buy' یا 'sell')
            is_reversed: آیا معامله Reversed است؟
        
        Returns:
            dict: {'new_sl': float, 'new_tp': float, 'action': str, 'reason': str}
        """
        # اگر نباید ML اعمال شود، همان مقادیر اولیه را برگردان
        if not self.should_apply_ml(is_reversed):
            return {
                'new_sl': sl,
                'new_tp': tp,
                'action': 'NO_CHANGE',
                'reason': 'ML only for Reversed trades' if is_reversed else 'Aligned trade'
            }
        
        try:
            # استخراج فیچرها از بازار
            features = self._extract_features(entry, trade_type)
            
            if features is None:
                return {
                    'new_sl': sl,
                    'new_tp': tp,
                    'action': 'NO_CHANGE',
                    'reason': 'Feature extraction failed'
                }
            
            # پیش‌بینی احتمال برد
            win_prob = self._predict_win_probability(features)
            
            # محاسبه فاصله SL و TP
            if trade_type.lower() == 'buy':
                sl_distance = entry - sl
                tp_distance = tp - entry
            else:
                sl_distance = sl - entry
                tp_distance = entry - tp
            
            # استراتژی ML: threshold 0.51
            if win_prob >= 0.51:
                # احتمال برد بالاتر → TP را افزایش می‌دهیم
                new_tp_distance = tp_distance * 1.5
                new_sl_distance = sl_distance * 0.7
                action = 'EXTEND_TP_TIGHTEN_SL'
                reason = f'High win probability ({win_prob:.1%})'
            else:
                # احتمال برد پایین‌تر → TP را کاهش می‌دهیم (خروج زودتر)
                new_tp_distance = tp_distance * 0.8
                new_sl_distance = sl_distance
                action = 'REDUCE_TP_EARLY_EXIT'
                reason = f'Lower win probability ({win_prob:.1%})'
            
            # محاسبه SL و TP جدید
            if trade_type.lower() == 'buy':
                new_sl = entry - new_sl_distance
                new_tp = entry + new_tp_distance
            else:
                new_sl = entry + new_sl_distance
                new_tp = entry - new_tp_distance
            
            return {
                'new_sl': new_sl,
                'new_tp': new_tp,
                'action': action,
                'reason': reason,
                'win_probability': win_prob
            }
            
        except Exception as e:
            print(f"⚠️ ML adjustment failed: {e}")
            return {
                'new_sl': sl,
                'new_tp': tp,
                'action': 'NO_CHANGE',
                'reason': f'Error: {str(e)}'
            }
    
    def _extract_features(self, entry_price, trade_type):
        """
        استخراج 21 فیچر از بازار
        
        Args:
            entry_price: قیمت ورود
            trade_type: نوع معامله
        
        Returns:
            np.array یا None
        """
        try:
            # گرفتن 60 کندل M1 اخیر (1 ساعت)
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 60)
            
            if rates is None or len(rates) < 10:
                return None
            
            # تبدیل به DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            close_prices = df['close'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            # محاسبه فیچرها (21 فیچر)
            features = {}
            
            # 1. نوسانات
            features['volatility'] = np.std(close_prices) / np.mean(close_prices) * 100
            
            # 2. RSI
            delta = np.diff(close_prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
            avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # 3. مومنتوم
            features['momentum'] = (close_prices[-1] - close_prices[0]) / close_prices[0] * 100
            
            # 4. قدرت روند
            ma_short = np.mean(close_prices[-10:])
            ma_long = np.mean(close_prices[-30:]) if len(close_prices) >= 30 else np.mean(close_prices)
            features['trend_strength'] = (ma_short - ma_long) / ma_long * 100
            
            # 5-6. فاصله از میانگین متحرک
            features['ma_distance_short'] = (entry_price - ma_short) / ma_short * 100
            features['ma_distance_long'] = (entry_price - ma_long) / ma_long * 100
            
            # 7-8. فاصله از سطوح حمایت/مقاومت
            recent_high = np.max(high_prices)
            recent_low = np.min(low_prices)
            features['support_distance'] = (entry_price - recent_low) / recent_low * 100
            features['resistance_distance'] = (recent_high - entry_price) / recent_high * 100
            
            # 9. قدرت شکست
            price_range = recent_high - recent_low
            features['breakout_strength'] = (entry_price - recent_low) / price_range if price_range != 0 else 0.5
            
            # 10. نسبت حجم (ساده‌سازی شده)
            features['volume_ratio'] = df['tick_volume'].iloc[-1] / df['tick_volume'].mean()
            
            # 11. موقعیت قیمت
            features['price_position'] = (entry_price - recent_low) / (recent_high - recent_low) if (recent_high - recent_low) != 0 else 0.5
            
            # 12-13. نسبت بدنه و سایه کندل
            last_candle = df.iloc[-1]
            body = abs(last_candle['close'] - last_candle['open'])
            total_range = last_candle['high'] - last_candle['low']
            features['candle_body_ratio'] = body / total_range if total_range != 0 else 0
            features['candle_wick_ratio'] = 1 - features['candle_body_ratio']
            
            # 14. نسبت High/Low اخیر
            features['recent_high_low_ratio'] = recent_high / recent_low if recent_low != 0 else 1
            
            # 15-16. زمان روز و روز هفته
            current_time = datetime.now()
            features['time_of_day'] = current_time.hour + current_time.minute / 60
            features['day_of_week'] = current_time.weekday()
            
            # 17. نوسانات ساعتی
            features['hour_volatility'] = np.std(close_prices[-12:]) if len(close_prices) >= 12 else features['volatility']
            
            # 18. شتاب قیمت
            if len(close_prices) >= 3:
                features['price_acceleration'] = (close_prices[-1] - close_prices[-2]) - (close_prices[-2] - close_prices[-3])
            else:
                features['price_acceleration'] = 0
            
            # 19. الگوی بازگشتی
            features['reversal_pattern'] = 1 if (close_prices[-1] > close_prices[-2] and close_prices[-2] < close_prices[-3]) else 0
            
            # 20. امتیاز تثبیت
            recent_std = np.std(close_prices[-5:])
            features['consolidation_score'] = 1 - min(recent_std / features['volatility'], 1) if features['volatility'] != 0 else 0
            
            # 21. ثبات روند
            price_changes = np.diff(close_prices[-10:])
            features['trend_consistency'] = len([x for x in price_changes if x * price_changes[-1] > 0]) / len(price_changes) if len(price_changes) > 0 else 0
            
            # ترتیب فیچرها (مطابق با مدل آموزش‌دیده)
            feature_names = ['volatility', 'rsi', 'momentum', 'trend_strength', 
                            'ma_distance_short', 'ma_distance_long', 'support_distance', 
                            'resistance_distance', 'breakout_strength', 'volume_ratio',
                            'price_position', 'candle_body_ratio', 'candle_wick_ratio',
                            'recent_high_low_ratio', 'time_of_day', 'day_of_week',
                            'hour_volatility', 'price_acceleration', 'reversal_pattern',
                            'consolidation_score', 'trend_consistency']
            
            feature_array = np.array([features.get(f, 0.0) for f in feature_names])
            
            # جایگزینی NaN و Inf
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return feature_array.reshape(1, -1)
            
        except Exception as e:
            print(f"⚠️ Feature extraction error: {e}")
            return None
    
    def _predict_win_probability(self, features):
        """
        پیش‌بینی احتمال برد
        
        Args:
            features: آرایه فیچرها
        
        Returns:
            float: احتمال برد (0-1)
        """
        try:
            # Normalize
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # پیش‌بینی
            prob = self.model.predict_proba(features)[0][1]
            
            return prob
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return 0.5  # احتمال خنثی


# نمونه استفاده:
if __name__ == "__main__":
    # تست
    ml_manager = MLPositionManager(symbol='GBPUSD')
    
    # شبیه‌سازی یک معامله Reversed
    result = ml_manager.adjust_sl_tp(
        entry=1.27000,
        sl=1.26950,
        tp=1.27100,
        trade_type='buy',
        is_reversed=True
    )
    
    print(f"\n📊 ML Decision:")
    print(f"   Action: {result['action']}")
    print(f"   New SL: {result['new_sl']:.5f}")
    print(f"   New TP: {result['new_tp']:.5f}")
    print(f"   Reason: {result['reason']}")
    if 'win_probability' in result:
        print(f"   Win Probability: {result['win_probability']:.1%}")
