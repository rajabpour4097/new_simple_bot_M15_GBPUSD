"""
تست نهایی ربات قبل از اجرای واقعی
این اسکریپت همه بخش‌ها را بررسی می‌کند
"""

import MetaTrader5 as mt5
from datetime import datetime
import sys

def test_imports():
    """تست همه import ها"""
    print("\n" + "="*60)
    print("📦 تست Import ها")
    print("="*60)
    
    errors = []
    
    try:
        from fibo_calculate import fibonacci_retracement
        print("✅ fibo_calculate")
    except Exception as e:
        errors.append(f"fibo_calculate: {e}")
        print(f"❌ fibo_calculate: {e}")
    
    try:
        from get_legs import get_legs
        print("✅ get_legs")
    except Exception as e:
        errors.append(f"get_legs: {e}")
        print(f"❌ get_legs: {e}")
    
    try:
        from mt5_connector import MT5Connector
        print("✅ MT5Connector")
    except Exception as e:
        errors.append(f"MT5Connector: {e}")
        print(f"❌ MT5Connector: {e}")
    
    try:
        from swing import get_swing_points
        print("✅ swing")
    except Exception as e:
        errors.append(f"swing: {e}")
        print(f"❌ swing: {e}")
    
    try:
        from utils import BotState
        print("✅ utils")
    except Exception as e:
        errors.append(f"utils: {e}")
        print(f"❌ utils: {e}")
    
    try:
        from save_file import log
        print("✅ save_file")
    except Exception as e:
        errors.append(f"save_file: {e}")
        print(f"❌ save_file: {e}")
    
    try:
        from metatrader5_config import MT5_CONFIG, TRADING_CONFIG
        print("✅ metatrader5_config")
    except Exception as e:
        errors.append(f"metatrader5_config: {e}")
        print(f"❌ metatrader5_config: {e}")
    
    try:
        from email_notifier import send_trade_email_async
        print("✅ email_notifier")
    except Exception as e:
        errors.append(f"email_notifier: {e}")
        print(f"❌ email_notifier: {e}")
    
    try:
        from analytics.hooks import log_signal
        print("✅ analytics.hooks")
    except Exception as e:
        errors.append(f"analytics.hooks: {e}")
        print(f"❌ analytics.hooks: {e}")
    
    try:
        from m15_filter_strategy import apply_m15_filter, format_m15_email_info
        print("✅ m15_filter_strategy")
    except Exception as e:
        errors.append(f"m15_filter_strategy: {e}")
        print(f"❌ m15_filter_strategy: {e}")
    
    return len(errors) == 0


def test_mt5_connection():
    """تست اتصال MT5"""
    print("\n" + "="*60)
    print("🔌 تست اتصال MT5")
    print("="*60)
    
    if not mt5.initialize():
        print("❌ MT5 initialize failed")
        print(f"   Error: {mt5.last_error()}")
        return False
    
    print("✅ MT5 initialized")
    
    # Account info
    acc = mt5.account_info()
    if acc:
        print(f"✅ Account: {acc.login}")
        print(f"   Balance: ${acc.balance:.2f}")
        print(f"   Server: {acc.server}")
    else:
        print("❌ Could not get account info")
        return False
    
    # Terminal info
    term = mt5.terminal_info()
    if term:
        print(f"✅ Terminal: AutoTrading = {term.trade_allowed}")
    
    return True


def test_symbol_info():
    """تست اطلاعات نماد"""
    print("\n" + "="*60)
    print("📊 تست اطلاعات نماد EURUSD")
    print("="*60)
    
    info = mt5.symbol_info("EURUSD")
    if not info:
        print("❌ Could not get symbol info")
        return False
    
    print(f"✅ Symbol: EURUSD")
    print(f"   Digits: {info.digits}")
    print(f"   Point: {info.point}")
    print(f"   Spread: {info.spread}")
    print(f"   Trade mode: {info.trade_mode}")
    print(f"   Volume min: {info.volume_min}")
    print(f"   Volume max: {info.volume_max}")
    
    tick = mt5.symbol_info_tick("EURUSD")
    if tick:
        print(f"✅ Current prices:")
        print(f"   Bid: {tick.bid}")
        print(f"   Ask: {tick.ask}")
        print(f"   Spread: {(tick.ask - tick.bid) * 10000:.1f} pips")
    
    return True


def test_m15_filter():
    """تست M15 filter"""
    print("\n" + "="*60)
    print("🔍 تست M15 Filter Strategy")
    print("="*60)
    
    from m15_filter_strategy import apply_m15_filter, get_last_completed_m15_candle
    
    # Get M15 candle
    candle = get_last_completed_m15_candle("EURUSD")
    if not candle:
        print("❌ Could not get M15 candle")
        return False
    
    print(f"✅ Last M15 Candle:")
    print(f"   Time: {candle['time']}")
    print(f"   Direction: {candle['direction']}")
    print(f"   Body Ratio: {candle['body_ratio']:.1f}%")
    print(f"   O={candle['open']:.5f} H={candle['high']:.5f} L={candle['low']:.5f} C={candle['close']:.5f}")
    
    # Test filter
    tick = mt5.symbol_info_tick("EURUSD")
    entry = tick.ask
    sl = entry - 0.0020  # 20 pips
    
    print(f"\n📋 Testing filter with BUY signal:")
    print(f"   Entry: {entry:.5f}")
    print(f"   Original SL: {sl:.5f}")
    
    action, reason, final_sl, final_tp, final_dir, m15_info = apply_m15_filter(
        signal_direction='buy',
        entry_price=entry,
        original_sl=sl,
        win_ratio=2.0,
        symbol='EURUSD'
    )
    
    print(f"\n📋 Filter Result:")
    print(f"   Action: {action}")
    print(f"   Reason: {reason}")
    
    if action == "EXECUTE_ORIGINAL":
        print(f"   ✅ Would execute: BUY (original)")
        print(f"   SL: {final_sl:.5f}, TP: {final_tp:.5f}")
    elif action == "EXECUTE_REVERSED":
        print(f"   🔄 Would execute: {final_dir.upper()} (reversed)")
        print(f"   SL: {final_sl:.5f}, TP: {final_tp:.5f}")
    else:
        print(f"   ❌ Would REJECT signal")
    
    return True


def test_config():
    """تست تنظیمات"""
    print("\n" + "="*60)
    print("⚙️ تست تنظیمات")
    print("="*60)
    
    from metatrader5_config import MT5_CONFIG, TRADING_CONFIG
    
    print(f"MT5_CONFIG:")
    print(f"   Symbol: {MT5_CONFIG['symbol']}")
    print(f"   Lot Size: {MT5_CONFIG['lot_size']}")
    print(f"   Win Ratio: {MT5_CONFIG['win_ratio']}")
    print(f"   Risk %: {MT5_CONFIG['risk_percent'] * 100}%")
    print(f"   Trading Hours: {MT5_CONFIG['trading_hours']['start']} - {MT5_CONFIG['trading_hours']['end']}")
    
    print(f"\nTRADING_CONFIG:")
    print(f"   Prevent Multiple Positions: {TRADING_CONFIG.get('prevent_multiple_positions', True)}")
    print(f"   Position Check Mode: {TRADING_CONFIG.get('position_check_mode', 'all')}")
    
    return True


def run_all_tests():
    """اجرای همه تست‌ها"""
    print("\n" + "🚀 "*20)
    print("      تست نهایی ربات - قبل از اجرای واقعی")
    print("🚀 "*20)
    print(f"Time: {datetime.now()}")
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    
    # Test 2: MT5 Connection
    results['mt5'] = test_mt5_connection()
    
    if results['mt5']:
        # Test 3: Symbol Info
        results['symbol'] = test_symbol_info()
        
        # Test 4: M15 Filter
        results['m15_filter'] = test_m15_filter()
        
        # Test 5: Config
        results['config'] = test_config()
        
        mt5.shutdown()
    
    # Summary
    print("\n" + "="*60)
    print("📋 خلاصه نتایج")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 همه تست‌ها موفق! ربات آماده اجراست.")
    else:
        print("⚠️ برخی تست‌ها ناموفق بودند. لطفاً بررسی کنید.")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
