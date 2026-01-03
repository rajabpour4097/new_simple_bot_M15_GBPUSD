"""
M15 Filter Strategy (S2: M15 Reverse >50%)

این ماژول استراتژی S2 را پیاده‌سازی می‌کند:
- اگر کندل M15 موافق روند سیگنال باشد → ورود عادی
- اگر کندل M15 مخالف روند باشد و قدرت بدنه > 50% → پوزیشن معکوس
- اگر کندل M15 مخالف روند باشد و قدرت بدنه <= 50% → رد سیگنال

نتایج تست:
2024: +537R با 75.4% win rate
2025: +758R با 73.4% win rate
"""

import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
from save_file import log as original_log
import inspect
import os


def log(message: str, color: str | None = None, save_to_file: bool = True):
    """Wrapper برای log با prefix"""
    try:
        frame = inspect.currentframe()
        caller = frame.f_back if frame else None
        lineno = getattr(caller, 'f_lineno', None)
        func = getattr(caller, 'f_code', None)
        fname = getattr(func, 'co_filename', None) if func else None
        funcname = getattr(func, 'co_name', None) if func else None
        base = os.path.basename(fname) if fname else 'unknown'
        prefix = f"[{base}:{funcname}:{lineno}] "
        return original_log(prefix + str(message), color=color, save_to_file=save_to_file)
    except Exception:
        return original_log(message, color=color, save_to_file=save_to_file)


def get_last_completed_m15_candle(symbol: str) -> Optional[Dict]:
    """
    دریافت آخرین کندل M15 تکمیل‌شده (نه کندل در حال تشکیل)
    
    Returns:
        dict با کلیدهای: time, open, high, low, close, direction, body_ratio
        یا None در صورت خطا
    """
    try:
        # دریافت 2 کندل آخر M15 (کندل آخر در حال تشکیل است)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 2)
        
        if rates is None or len(rates) < 2:
            log(f"❌ Could not get M15 candles for {symbol}", color='red')
            return None
        
        # کندل قبلی (تکمیل‌شده) - ایندکس 0
        candle = rates[0]
        
        open_price = float(candle['open'])
        high_price = float(candle['high'])
        low_price = float(candle['low'])
        close_price = float(candle['close'])
        candle_time = datetime.fromtimestamp(candle['time'])
        
        # محاسبه جهت کندل
        if close_price > open_price:
            direction = 'bullish'
        elif close_price < open_price:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # محاسبه نسبت بدنه
        candle_range = high_price - low_price
        body_size = abs(close_price - open_price)
        
        if candle_range > 0:
            body_ratio = (body_size / candle_range) * 100
        else:
            body_ratio = 0
        
        return {
            'time': candle_time,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'direction': direction,
            'body_ratio': body_ratio,
            'range': candle_range,
            'body_size': body_size
        }
        
    except Exception as e:
        log(f"❌ Error getting M15 candle: {e}", color='red')
        return None


def apply_m15_filter(
    signal_direction: str,  # 'buy' یا 'sell'
    entry_price: float,
    original_sl: float,
    win_ratio: float,
    symbol: str
) -> Tuple[str, str, float, float, float, Dict]:
    """
    اعمال فیلتر استراتژی S2 روی سیگنال
    
    Args:
        signal_direction: جهت سیگنال اصلی ('buy' یا 'sell')
        entry_price: قیمت ورود
        original_sl: استاپ‌لاس اصلی (fib 1.0)
        win_ratio: نسبت RR (مثلاً 2 برای 1:2)
        symbol: نماد معاملاتی
    
    Returns:
        Tuple[action, reason, final_sl, final_tp, final_direction, m15_info]
        - action: 'EXECUTE_ORIGINAL', 'EXECUTE_REVERSED', 'REJECT'
        - reason: دلیل تصمیم
        - final_sl: استاپ‌لاس نهایی
        - final_tp: تیک‌پرافیت نهایی
        - final_direction: جهت نهایی پوزیشن ('buy' یا 'sell')
        - m15_info: اطلاعات کندل M15
    """
    
    # دریافت کندل M15
    m15 = get_last_completed_m15_candle(symbol)
    
    if m15 is None:
        log(f"⚠️ Could not get M15 candle - executing original signal", color='yellow')
        # در صورت عدم دسترسی به M15، سیگنال اصلی اجرا شود
        stop_distance = abs(entry_price - original_sl)
        if signal_direction == 'buy':
            final_tp = entry_price + (stop_distance * win_ratio)
        else:
            final_tp = entry_price - (stop_distance * win_ratio)
        
        return ('EXECUTE_ORIGINAL', 'M15 data unavailable', original_sl, final_tp, signal_direction, {})
    
    log(f"📊 M15 Candle: time={m15['time']} dir={m15['direction']} body={m15['body_ratio']:.1f}%", color='cyan')
    
    # تعیین جهت مورد انتظار M15 (موافق با سیگنال)
    expected_m15_direction = 'bullish' if signal_direction == 'buy' else 'bearish'
    
    # بررسی تطابق جهت
    is_aligned = (m15['direction'] == expected_m15_direction)
    
    if is_aligned:
        # ✅ موافق روند - اجرای سیگنال اصلی
        log(f"✅ M15 ALIGNED: {m15['direction']} matches {signal_direction} signal", color='green')
        
        stop_distance = abs(entry_price - original_sl)
        if signal_direction == 'buy':
            final_tp = entry_price + (stop_distance * win_ratio)
        else:
            final_tp = entry_price - (stop_distance * win_ratio)
        
        return (
            'EXECUTE_ORIGINAL',
            f"M15 aligned ({m15['direction']}, body={m15['body_ratio']:.1f}%)",
            original_sl,
            final_tp,
            signal_direction,
            m15
        )
    
    else:
        # مخالف روند - بررسی قدرت بدنه
        if m15['body_ratio'] > 50:
            # ✅ قدرت بالا - پوزیشن معکوس
            log(f"🔄 M15 REVERSE: {m15['direction']} with strong body {m15['body_ratio']:.1f}% > 50%", color='blue')
            
            # معکوس کردن جهت
            reversed_direction = 'sell' if signal_direction == 'buy' else 'buy'
            
            # محاسبه SL و TP معکوس
            # SL: همان فاصله ولی در جهت معکوس
            stop_distance = abs(entry_price - original_sl)
            
            if reversed_direction == 'buy':
                # سیگنال اصلی SELL بود، حالا BUY می‌گیریم
                # SL زیر entry
                reversed_sl = entry_price - stop_distance
                reversed_tp = entry_price + (stop_distance * win_ratio)
            else:
                # سیگنال اصلی BUY بود، حالا SELL می‌گیریم
                # SL بالای entry
                reversed_sl = entry_price + stop_distance
                reversed_tp = entry_price - (stop_distance * win_ratio)
            
            return (
                'EXECUTE_REVERSED',
                f"M15 opposite ({m15['direction']}) with body={m15['body_ratio']:.1f}% > 50%",
                reversed_sl,
                reversed_tp,
                reversed_direction,
                m15
            )
        
        else:
            # ❌ قدرت پایین - رد سیگنال
            log(f"❌ M15 REJECT: {m15['direction']} with weak body {m15['body_ratio']:.1f}% <= 50%", color='red')
            
            return (
                'REJECT',
                f"M15 opposite ({m15['direction']}) with weak body={m15['body_ratio']:.1f}% <= 50%",
                0,
                0,
                '',
                m15
            )


def format_m15_email_info(action: str, reason: str, m15_info: Dict, 
                          original_direction: str, final_direction: str) -> str:
    """
    فرمت کردن اطلاعات M15 برای ایمیل
    """
    if not m15_info:
        return "M15 Info: Not available\n"
    
    status_emoji = {
        'EXECUTE_ORIGINAL': '✅',
        'EXECUTE_REVERSED': '🔄',
        'REJECT': '❌'
    }.get(action, '❓')
    
    lines = [
        f"\n📊 M15 Filter Analysis:",
        f"   Status: {status_emoji} {action}",
        f"   Reason: {reason}",
        f"   M15 Candle Time: {m15_info.get('time', 'N/A')}",
        f"   M15 Direction: {m15_info.get('direction', 'N/A')}",
        f"   M15 Body Strength: {m15_info.get('body_ratio', 0):.1f}%",
        f"   Original Signal: {original_direction.upper()}",
    ]
    
    if action == 'EXECUTE_REVERSED':
        lines.append(f"   Final Direction: {final_direction.upper()} (REVERSED)")
    elif action == 'EXECUTE_ORIGINAL':
        lines.append(f"   Final Direction: {final_direction.upper()} (ALIGNED)")
    
    return '\n'.join(lines) + '\n'


# تست ماژول
if __name__ == '__main__':
    # تست اتصال به MT5
    if not mt5.initialize():
        print("Failed to initialize MT5")
    else:
        print("MT5 initialized successfully")
        
        # تست دریافت کندل M15
        candle = get_last_completed_m15_candle('EURUSD')
        if candle:
            print(f"\nLast M15 candle:")
            print(f"  Time: {candle['time']}")
            print(f"  Direction: {candle['direction']}")
            print(f"  Body ratio: {candle['body_ratio']:.1f}%")
            print(f"  O={candle['open']}, H={candle['high']}, L={candle['low']}, C={candle['close']}")
        
        # تست فیلتر
        print("\n--- Testing filter for BUY signal ---")
        result = apply_m15_filter(
            signal_direction='buy',
            entry_price=1.04500,
            original_sl=1.04300,
            win_ratio=2.0,
            symbol='EURUSD'
        )
        print(f"Action: {result[0]}")
        print(f"Reason: {result[1]}")
        print(f"SL: {result[2]}, TP: {result[3]}")
        print(f"Direction: {result[4]}")
        
        mt5.shutdown()
