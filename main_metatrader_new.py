import MetaTrader5 as mt5
from datetime import datetime
from fibo_calculate import fibonacci_retracement
import numpy as np
import pandas as pd
from time import sleep
from colorama import init, Fore
from get_legs import get_legs
from mt5_connector import MT5Connector
from swing import get_swing_points
from utils import BotState
from save_file import log
import inspect, os
from metatrader5_config import MT5_CONFIG, TRADING_CONFIG, DYNAMIC_RISK_CONFIG
from email_notifier import send_trade_email_async
from analytics.hooks import log_signal, log_position_event
from m15_filter_strategy import apply_m15_filter, format_m15_email_info
from ml_position_manager import MLPositionManager
from ml_position_manager import MLPositionManager



def main():
    # راه‌اندازی MT5 و colorama
    init(autoreset=True)
    mt5_conn = MT5Connector()

    if not mt5_conn.initialize():
        print("❌ Failed to connect to MT5")
        return

    # Initial state با تنظیمات - مطابق main_saver_copy2.py
    state = BotState()
    state.reset()

    start_index = 0
    win_ratio = MT5_CONFIG['win_ratio']
    threshold = TRADING_CONFIG['threshold']
    window_size = TRADING_CONFIG['window_size']
    min_swing_size = TRADING_CONFIG['min_swing_size']

    i = 1
    f = 0
    position_open = False
    last_swing_type = None

    print(f"🚀 MT5 Trading Bot Started...")
    print(f"📊 Config: Symbol={MT5_CONFIG['symbol']}, Lot={MT5_CONFIG['lot_size']}, Win Ratio={win_ratio}")
    print(f"⏰ Trading Hours (Iran): {MT5_CONFIG['trading_hours']['start']} - {MT5_CONFIG['trading_hours']['end']}")
    print(f"🇮🇷 Current Iran Time: {mt5_conn.get_iran_time().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # راه‌اندازی ML Position Manager
    ml_manager = MLPositionManager(symbol=MT5_CONFIG['symbol'])
    print("-" * 50)
    
    # نمایش تنظیمات مدیریت پوزیشن
    prevent_multiple = TRADING_CONFIG.get('prevent_multiple_positions', True)
    check_mode = TRADING_CONFIG.get('position_check_mode', 'all')
    print(f"🔒 Position Management: Multiple positions prevention = {prevent_multiple}")
    if prevent_multiple:
        print(f"🔍 Check Mode: {check_mode} ({'All positions' if check_mode == 'all' else 'Only conflicting positions'})")

    # در ابتدای main loop بعد از initialize
    print("🔍 Checking symbol properties...")
    mt5_conn.check_symbol_properties()
    print("🔍 Testing broker filling modes...")
    mt5_conn.test_filling_modes()
    mt5_conn.check_trading_limits()
    print("🔍 Checking account permissions...")
    mt5_conn.check_account_trading_permissions()
    print("🔍 Checking market state...")
    mt5_conn.check_market_state()
    print("-" * 50)

    # --- Contextual logging wrapper: prefix logs with file:function:line ---
    # Import original log function with alias to avoid conflict
    from save_file import log as original_log
    
    def log(message: str, color: str | None = None, save_to_file: bool = True):
        try:
            frame = inspect.currentframe()
            # Walk back to the caller outside this wrapper
            caller = frame.f_back if frame else None
            lineno = getattr(caller, 'f_lineno', None)
            func = getattr(caller, 'f_code', None)
            fname = getattr(func, 'co_filename', None) if func else None
            funcname = getattr(func, 'co_name', None) if func else None
            base = os.path.basename(fname) if fname else 'unknown'
            prefix = f"[{base}:{funcname}:{lineno}] "
            return original_log(prefix + str(message), color=color, save_to_file=save_to_file)
        except Exception:
            # Fallback to original log if anything goes wrong
            return original_log(message, color=color, save_to_file=save_to_file)

    # اضافه کردن متغیر برای ذخیره آخرین داده
    last_data_time = None
    wait_count = 0
    max_wait_cycles = 120  # پس از 60 ثانیه (120 * 0.5) اجبار به پردازش
    # نگهداری وضعیت قبلی قابلیت معامله برای ریست در انتهای ساعات ترید
    last_can_trade_state = None

    # بعد از تعریف متغیرها در main()
    def reset_state_and_window():
        nonlocal start_index
        state.reset()
        start_index = max(0, len(cache_data) - window_size)
        log(f'Reset state -> new start_index={start_index} (slice len={len(cache_data.iloc[start_index:])})', color='magenta')
    
    # حالت‌های مدیریت پوزیشن
    position_states = {}  # ticket -> {'entry':..., 'risk':..., 'direction':..., 'done_stages':set(), 'base_tp_R':float, 'commission_locked':False}

    def _digits():
        info = mt5.symbol_info(MT5_CONFIG['symbol'])
        return info.digits if info else 5

    def _round(p):
        return float(f"{p:.{_digits()}f}")

    def has_open_positions():
        """بررسی وجود پوزیشن‌های باز"""
        positions = mt5_conn.get_positions()
        return positions is not None and len(positions) > 0

    def has_conflicting_positions(intended_direction):
        """بررسی وجود پوزیشن‌های مخالف با جهت مورد نظر
        intended_direction: 'buy' یا 'sell'
        """
        positions = mt5_conn.get_positions()
        if not positions:
            return False
        
        for pos in positions:
            if intended_direction == 'buy' and pos.type == mt5.POSITION_TYPE_SELL:
                return True
            elif intended_direction == 'sell' and pos.type == mt5.POSITION_TYPE_BUY:
                return True
        return False

    def log_open_positions():
        """نمایش جزئیات پوزیشن‌های باز"""
        positions = mt5_conn.get_positions()
        if not positions:
            return
        log(f"📊 Open positions count: {len(positions)}", color='cyan')
        for pos in positions:
            pos_type = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
            log(f"   Ticket={pos.ticket} | Type={pos_type} | Volume={pos.volume} | Entry={pos.price_open} | Profit={pos.profit:.2f}", color='cyan')

    def get_positions_summary():
        """دریافت خلاصه‌ای از پوزیشن‌های باز برای ایمیل"""
        positions = mt5_conn.get_positions()
        if not positions:
            return "No open positions"
        
        summary = []
        for pos in positions:
            pos_type = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
            summary.append(f"   - Ticket: {pos.ticket} | Type: {pos_type} | Volume: {pos.volume} | Entry: {pos.price_open} | Profit: {pos.profit:.2f}")
        
        return f"{len(positions)} open position(s):\n" + "\n".join(summary)

    def register_position(pos):
        # محاسبه R (ریسک اولیه)
        risk = abs(pos.price_open - pos.sl) if pos.sl else None
        if not risk or risk == 0:
            return
        position_states[pos.ticket] = {
            'entry': pos.price_open,
            'risk': risk,
            'direction': 'buy' if pos.type == mt5.POSITION_TYPE_BUY else 'sell',
            'done_stages': set(),
            'base_tp_R': DYNAMIC_RISK_CONFIG.get('base_tp_R', 2),
            'commission_locked': False
        }
        # رویداد ثبت پوزیشن
        try:
            log_position_event(
                symbol=MT5_CONFIG['symbol'],
                ticket=pos.ticket,
                event='open',
                direction=position_states[pos.ticket]['direction'],
                entry=pos.price_open,
                current_price=pos.price_open,
                sl=pos.sl,
                tp=pos.tp,
                profit_R=0.0,
                stage=0,
                risk_abs=risk,
                locked_R=None,
                volume=pos.volume,
                note='position registered'
            )
        except Exception:
            pass

    def manage_open_positions():
        if not DYNAMIC_RISK_CONFIG.get('enable'):
            return
        positions = mt5_conn.get_positions()
        if not positions:
            return
        tick = mt5.symbol_info_tick(MT5_CONFIG['symbol'])
        if not tick:
            return
        stages_cfg = DYNAMIC_RISK_CONFIG.get('stages', [])
        for pos in positions:
            if pos.ticket not in position_states:
                register_position(pos)
            st = position_states.get(pos.ticket)
            if not st:
                continue
            entry = st['entry']
            risk = st['risk']
            direction = st['direction']
            cur_price = tick.bid if direction == 'buy' else tick.ask
            # profit in price
            if direction == 'buy':
                price_profit = cur_price - entry
            else:
                price_profit = entry - cur_price
            profit_R = price_profit / risk if risk else 0.0
            modified_any = False

            # محاسبه ارزش پولی 1R تقریبی (بدون اسپرد) برای تبدیل کامیشن به R:
            # risk_abs_price = risk (فاصله قیمتی) * volume * contract ارزش واقعی - ساده‌سازی: فقط نسبت بر اساس فاصله قیمتی.
            # برای دقت بیشتر باید tick_value استفاده شود؛ اینجا ساده نگه می‌داریم.

            # عبور از مراحل R-based
            for stage_cfg in stages_cfg:
                sid = stage_cfg.get('id')
                if sid in st['done_stages']:
                    continue
                new_sl = None
                new_tp = None
                event_name = None
                locked_R = None

                # R-based stage
                trigger_R = stage_cfg.get('trigger_R')
                if trigger_R is not None and profit_R >= trigger_R:
                    sl_lock_R = stage_cfg.get('sl_lock_R', trigger_R)
                    tp_R = stage_cfg.get('tp_R')
                    # SL placement
                    if direction == 'buy':
                        new_sl = entry + sl_lock_R * risk
                        if tp_R:
                            new_tp = entry + tp_R * risk
                    else:
                        new_sl = entry - sl_lock_R * risk
                        if tp_R:
                            new_tp = entry - tp_R * risk
                    event_name = sid
                    locked_R = sl_lock_R

                if new_sl is not None:
                    # Round
                    new_sl_r = _round(new_sl)
                    new_tp_r = _round(new_tp) if new_tp is not None else pos.tp
                    # Apply only if improves
                    apply = False
                    if direction == 'buy' and new_sl_r > pos.sl:
                        apply = True
                    if direction == 'sell' and new_sl_r < pos.sl:
                        apply = True
                    if apply:
                        res = mt5_conn.modify_sl_tp(pos.ticket, new_sl=new_sl_r, new_tp=new_tp_r)
                        if res and getattr(res, 'retcode', None) == 10009:
                            st['done_stages'].add(sid)
                            modified_any = True
                            log(f'⚙️ Dynamic Risk Stage {sid} applied: ticket={pos.ticket} | Profit: {profit_R:.2f}R | SL: {new_sl_r} | TP: {new_tp_r}', color='cyan')
                            try:
                                log_position_event(
                                    symbol=MT5_CONFIG['symbol'],
                                    ticket=pos.ticket,
                                    event=event_name or sid,
                                    direction=direction,
                                    entry=entry,
                                    current_price=cur_price,
                                    sl=new_sl_r,
                                    tp=new_tp_r,
                                    profit_R=profit_R,
                                    stage=None,
                                    risk_abs=risk,
                                    locked_R=locked_R,
                                    volume=pos.volume,
                                    note=f'stage {sid} trigger'
                                )
                            except Exception:
                                pass
            if modified_any:
                position_states[pos.ticket] = st

    while True:
        try:
            # بررسی ساعات معاملاتی
            can_trade, trade_message = mt5_conn.can_trade()
            # اگر از حالت قابل معامله به غیرقابل معامله تغییر کرد => ریست کامل BotState
            try:
                if last_can_trade_state is True and not can_trade:
                    log("🧹 Trading hours ended -> resetting BotState to avoid stale context", color='magenta')
                    state.reset()
            except Exception:
                pass
            finally:
                last_can_trade_state = can_trade
            
            if not can_trade:
                log(f"⏰ {trade_message}", color='yellow', save_to_file=False)
                sleep(60)
                continue
            
            # دریافت داده از MT5
            cache_data = mt5_conn.get_historical_data(count=window_size)
            
            if cache_data is None:
                log("❌ Failed to get data from MT5", color='red')
                sleep(5)
                continue
                
            cache_data['status'] = np.where(cache_data['open'] > cache_data['close'], 'bearish', 'bullish')
            
            # بررسی تغییر داده - مشابه main_saver_copy2.py
            current_time = cache_data.index[-1]
            if last_data_time is None:
                log(f"🔄 First run - processing data from {current_time}", color='cyan')
                last_data_time = current_time
                process_data = True
                wait_count = 0
            elif current_time != last_data_time:
                log(f"📊 New data received: {current_time} (previous: {last_data_time})", color='cyan')
                last_data_time = current_time
                process_data = True
                wait_count = 0
            else:
                wait_count += 1
                if wait_count % 20 == 0:  # هر 10 ثانیه یک بار لاگ
                    log(f"⏳ Waiting for new data... Current: {current_time} (wait cycles: {wait_count})", color='yellow', save_to_file=False)
                
                # اگر خیلی زیاد انتظار کشیدیم، اجبار به پردازش (در صورت تست)
                if wait_count >= max_wait_cycles:
                    log(f"⚠️ Force processing after {wait_count} cycles without new data", color='magenta')
                    process_data = True
                    wait_count = 0
                else:
                    process_data = False
            
            if process_data:
                log((' ' * 80 + '\n') * 3)
                log(f'Log number {i}:', color='lightred_ex')
                log(f'📊 Processing {len(cache_data)} data points | Window: {window_size}', color='cyan')
                log(f'Current time: {cache_data.index[-1]}', color='yellow')
                log(f'Start index: {start_index}  value: {cache_data.iloc[0].timestamp}  end data: {cache_data.iloc[-2].timestamp}', color='yellow')
                log(f'len data: {len(cache_data)} ', color='yellow')
                log(f'Current data status: {cache_data.iloc[-1]["status"]} open: {cache_data.iloc[-1]["open"]} close: {cache_data.iloc[-1]["close"]} time: {cache_data.index[-1]}')
                log(f'Last data status: {cache_data.iloc[-2]["status"]} open: {cache_data.iloc[-2]["open"]} close: {cache_data.iloc[-2]["close"]} time: {cache_data.index[-2]}')
                log(f' ' * 80)
                i += 1
                
                legs = get_legs(cache_data)
                log(f'First len legs: {len(legs)}', color='green')
                log(f' ' * 80)

                if len(legs) > 2:
                    log(f'legs > 2', color='blue')
                    legs = legs[-3:]
                    log(f"{cache_data.loc[legs[0]['start']].name} {cache_data.loc[legs[0]['end']].name} "
                        f"{cache_data.loc[legs[1]['start']].name} {cache_data.loc[legs[1]['end']].name} "
                        f"{cache_data.loc[legs[2]['start']].name} {cache_data.loc[legs[2]['end']].name}", color='yellow')
                    swing_type, is_swing = get_swing_points(data=cache_data, legs=legs)


                    # log(f'legs[1][start]start_value: {legs[1]['start_value']}', color='green')
                    # log(f'legs[1][start]end_value: {legs[1]['end_value']}', color='green')
                    # log(f'legs[1] TEST: {legs[1]}', color='green')
                    # log(f'Test: cache_data.index[-1][close]: {cache_data.iloc[-1]['close']}', color='green')


                    # Phase 1 Initialization fib_levels or change by new fib
                    
                    if is_swing:
                        log(f"is_swing: {swing_type}")
                        if swing_type == 'bullish' and cache_data.iloc[-2]['close'] > legs[1]['start_value']:
                            state.reset()
                            state.fib_levels = fibonacci_retracement(start_price=legs[2]['end_value'], end_price=legs[2]['start_value'])
                            state.fib0_time = legs[2]['start']
                            state.fib1_time = legs[2]['end']
                            last_swing_type = swing_type
                            log(f"📈 New fibonacci created: fib1:{state.fib_levels['1.0']} time:{legs[2]['start']} - fib0.705:{state.fib_levels['0.705']} - fib0:{state.fib_levels['0.0']} time:{legs[2]['end']}", color='green')

                        elif swing_type == 'bearish' and cache_data.iloc[-2]['close'] < legs[1]['start_value']:
                            state.reset()
                            state.fib_levels = fibonacci_retracement(start_price=legs[2]['end_value'], end_price=legs[2]['start_value'])
                            state.fib0_time = legs[2]['start']
                            state.fib1_time = legs[2]['end']
                            last_swing_type = swing_type
                            log(f"📉 New fibonacci created: fib1:{state.fib_levels['1.0']} time:{legs[2]['start']} - fib0.705:{state.fib_levels['0.705']} - fib0:{state.fib_levels['0.0']} time:{legs[2]['end']}", color='green')

                    # Phase 2
                    if state.fib_levels:
                        log(f'📊 Phase 2', color='blue')
                        if last_swing_type == 'bullish':
                            if cache_data.iloc[-2]['high'] > state.fib_levels['0.0']:
                                state.fib_levels = fibonacci_retracement(start_price=cache_data.iloc[-2]['high'], end_price=state.fib_levels['1.0'])
                                state.fib0_time = cache_data.iloc[-2]['timestamp']
                                state.first_touch = False
                                state.first_touch_value = None
                                # Should it be reset???
                                log(f"📈 Updated fibonacci: fib1:{state.fib_levels['1.0']} - fib0.705:{state.fib_levels['0.705']} - fib0:{state.fib_levels['0.0']}", color='green')
                            elif cache_data.iloc[-2]['low'] < state.fib_levels['1.0']:
                                state.reset()
                                log(f"📈 Price dropped below fib1 on bullish and reset fib levels", color='red')
                            elif cache_data.iloc[-2]['low'] <= state.fib_levels['0.705']:
                                log(f"📈 Price touched fib0.705 on bullish -- cache_data status is {cache_data.iloc[-2]['status']}", color='red')
                                if not state.first_touch:
                                    state.first_touch_value = cache_data.iloc[-2]
                                    state.first_touch = True
                                    log(f"📈 First touch on bullish: {state.first_touch_value['timestamp']}  first touch status is {state.first_touch_value['status']}", color='green')
                                elif state.first_touch and not state.second_touch and cache_data.iloc[-2]['status'] != state.first_touch_value['status']:
                                    state.second_touch_value = cache_data.iloc[-2]
                                    state.second_touch = True
                                    log(f"📈 Second touch on bullish: {state.second_touch_value['timestamp']}  second touch status is {state.second_touch_value['status']}", color='green')

                        elif last_swing_type == 'bearish':
                            if cache_data.iloc[-2]['low'] < state.fib_levels['0.0']:
                                state.fib_levels = fibonacci_retracement(start_price=cache_data.iloc[-2]['low'], end_price=state.fib_levels['1.0'])
                                state.fib0_time = cache_data.iloc[-2]['timestamp']
                                state.first_touch = False
                                state.first_touch_value = None
                                # Should it be reset???
                                log(f"📉 Updated fibonacci: fib1:{state.fib_levels['1.0']} - fib0.705:{state.fib_levels['0.705']} - fib0:{state.fib_levels['0.0']}", color='green')
                            elif cache_data.iloc[-2]['high'] > state.fib_levels['1.0']:
                                state.reset()
                                log(f"📉 Price dropped below fib1 on bearish and reset fib levels", color='red')
                            elif cache_data.iloc[-2]['high'] >= state.fib_levels['0.705']:
                                log(f"📉 Price touched fib0.705 on bearish -- cache_data status is {cache_data.iloc[-2]['status']}", color='red')
                                if not state.first_touch:
                                    state.first_touch_value = cache_data.iloc[-2]
                                    state.first_touch = True
                                    log(f"📉 First touch on bearish: {state.first_touch_value['timestamp']}  first touch status is {state.first_touch_value['status']}", color='red')
                                elif state.first_touch and not state.second_touch and cache_data.iloc[-2]['status'] != state.first_touch_value['status']:
                                    state.second_touch_value = cache_data.iloc[-2]
                                    state.second_touch = True
                                    log(f"📉 Second touch on bearish: {state.second_touch_value['timestamp']}  second touch status is {state.second_touch_value['status']}", color='red')

                    elif not is_swing and not state.fib_levels:
                        pass

                if len(legs) < 3:
                    # Phase 3
                    if state.fib_levels:
                        log(f"📊 Phase 3", color='blue')
                        if last_swing_type == 'bullish':
                            if cache_data.iloc[-2]['high'] > state.fib_levels['0.0']:
                                state.fib_levels = fibonacci_retracement(start_price=cache_data.iloc[-2]['high'], end_price=state.fib_levels['1.0'])
                                state.fib0_time = cache_data.iloc[-2]['timestamp']
                                state.first_touch = False
                                state.first_touch_value = None
                                # Should it be reset???
                                log(f"📈 Updated fibonacci: fib1:{state.fib_levels['1.0']} - fib0.705:{state.fib_levels['0.705']} - fib0:{state.fib_levels['0.0']}", color='green')
                            elif cache_data.iloc[-2]['low'] < state.fib_levels['1.0']:
                                state.reset()
                                log(f"📈 Price dropped below fib1 on bullish and reset fib levels", color='red')
                            elif cache_data.iloc[-2]['low'] <= state.fib_levels['0.705']:
                                log(f"📈 Price touched fib0.705 on bullish -- cache_data status is {cache_data.iloc[-2]['status']}", color='red')
                                if not state.first_touch:
                                    state.first_touch = True
                                    state.first_touch_value = cache_data.iloc[-2]
                                    log(f"📈 First touch on bullish: {state.first_touch_value['timestamp']}  first touch status is {state.first_touch_value['status']}", color='green')
                                elif state.first_touch and not state.second_touch and cache_data.iloc[-2]['status'] != state.first_touch_value['status']:
                                    state.second_touch = True
                                    state.second_touch_value = cache_data.iloc[-2]
                                    log(f"📈 Second touch on bullish: {state.second_touch_value['timestamp']}  second touch status is {state.second_touch_value['status']}", color='green')

                        elif last_swing_type == 'bearish':
                            if cache_data.iloc[-2]['low'] < state.fib_levels['0.0']:
                                state.fib_levels = fibonacci_retracement(start_price=cache_data.iloc[-2]['low'], end_price=state.fib_levels['1.0'])
                                state.fib0_time = cache_data.iloc[-2]['timestamp']
                                state.first_touch = False
                                state.first_touch_value = None
                                # Should it be reset???
                                log(f"📉 Updated fibonacci: fib1:{state.fib_levels['1.0']} - fib0.705:{state.fib_levels['0.705']} - fib0:{state.fib_levels['0.0']}", color='green')
                            elif cache_data.iloc[-2]['high'] > state.fib_levels['1.0']:
                                state.reset()
                                log(f"📉 Price dropped below fib1 on bearish and reset fib levels", color='red')
                            elif cache_data.iloc[-2]['high'] >= state.fib_levels['0.705']:
                                log(f"📉 Price touched fib0.705 on bearish -- cache_data status is {cache_data.iloc[-2]['status']}", color='red')
                                if not state.first_touch:
                                    state.first_touch_value = cache_data.iloc[-2]
                                    state.first_touch = True
                                    log(f"📉 First touch on bearish: {state.first_touch_value['timestamp']}  first touch status is {state.first_touch_value['status']}", color='red')
                                elif state.first_touch and not state.second_touch and cache_data.iloc[-2]['status'] != state.first_touch_value['status']:
                                    state.second_touch_value = cache_data.iloc[-2]
                                    state.second_touch = True
                                    log(f"📉 Second touch on bearish: {state.second_touch_value['timestamp']}  second touch status is {state.second_touch_value['status']}", color='red')

                    if len(legs) == 2:
                        log(f'legs = 2', color='blue')
                        log(f'leg0: {legs[0]["start"]}, {legs[0]["end"]}, leg1: {legs[1]["start"]}, {legs[1]["end"]}', color='lightcyan_ex')
                    elif len(legs) == 1:
                        log(f'legs = 1', color='blue')
                        log(f'leg0: {legs[0]["start"]}, {legs[0]["end"]}', color='lightcyan_ex')
                
                # بخش معاملات - buy statement (مطابق منطق main_saver_copy2.py)
                if last_swing_type == 'bullish' and state.second_touch:
                    # بررسی پوزیشن‌های باز قبل از ایجاد سیگنال جدید (اگر فعال باشد)
                    if TRADING_CONFIG.get('prevent_multiple_positions', True):
                        check_mode = TRADING_CONFIG.get('position_check_mode', 'all')
                        should_skip = False
                        skip_reason = ""
                        
                        if check_mode == 'all' and has_open_positions():
                            log(f"🚫 Skip BUY signal: Position(s) already open (mode: all positions)", color='yellow')
                            should_skip = True
                            skip_reason = f"Position(s) already open (mode: {check_mode})"
                        elif check_mode == 'conflicting' and has_conflicting_positions('buy'):
                            log(f"🚫 Skip BUY signal: Conflicting SELL position(s) detected", color='yellow')
                            should_skip = True
                            skip_reason = "Conflicting SELL position(s) detected"
                        
                        if should_skip:
                            log_open_positions()
                            
                            # ارسال ایمیل اطلاع‌رسانی skip شدن سیگنال BUY
                            try:
                                positions_summary = get_positions_summary()
                                send_trade_email_async(
                                    subject=f"SIGNAL SKIPPED - BUY {MT5_CONFIG['symbol']}",
                                    body=(
                                        f"It's for 5000 dollars account.\n\n"
                                        f"🚫 TRADING SIGNAL SKIPPED 🚫\n\n"
                                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                        f"Symbol: {MT5_CONFIG['symbol']}\n"
                                        f"Signal Type: BUY (Bullish Swing)\n"
                                        f"Action: SKIPPED\n"
                                        f"Reason: {skip_reason}\n"
                                        f"Check Mode: {check_mode}\n\n"
                                        f"📊 Signal Details:\n"
                                        f"Entry Price Would Be: {buy_entry_price:.5f}\n"
                                        f"Stop Loss Would Be: {state.fib_levels.get('1.0', 'N/A'):.5f}\n\n"
                                        f"📈 Fibonacci Levels:\n"
                                        f"   fib 0.0 (resistance): {state.fib_levels.get('0.0', 'N/A'):.5f}\n"
                                        f"   fib 0.705 (entry zone): {state.fib_levels.get('0.705', 'N/A'):.5f}\n"
                                        f"   fib 1.0 (support/SL): {state.fib_levels.get('1.0', 'N/A'):.5f}\n\n"
                                        f"🔒 Current Open Positions:\n{positions_summary}\n"
                                    )
                                )
                                log(f"📧 Skip signal email sent for BUY signal", color='cyan')
                            except Exception as _e:
                                log(f'Skip signal email failed: {_e}', color='red')
                            
                            state.reset()
                            reset_state_and_window()
                            continue
                    
                    log(f"📈 Buy signal triggered", color='green')
                    last_tick = mt5.symbol_info_tick(MT5_CONFIG['symbol'])
                    buy_entry_price = last_tick.ask
                    log(f'Start long position income {cache_data.iloc[-1].name}', color='blue')
                    log(f'current_open_point (market ask): {buy_entry_price}', color='blue')
                    # ENTRY CONTEXT (BUY): fib snapshot + touches
                    try:
                        fib = state.fib_levels or {}
                        fib0_p = fib.get('0.0')
                        fib1_p = fib.get('1.0')
                        log(
                            f"ENTRY_CTX_BUY | fib0_time={state.fib0_time} value={fib0_p} | fib705={fib.get('0.705')} | fib09={fib.get('0.9')} | fib1_time={state.fib1_time} value={fib1_p}",
                            color='cyan'
                        )
                    except Exception:
                        pass

                    pip_size = _pip_size_for(MT5_CONFIG['symbol'])
                    two_pips = 2.0 * pip_size
                    min_dist = _min_stop_distance(MT5_CONFIG['symbol'])

                    # همیشه از fib 1.0 استفاده می‌کنیم
                    candidate_sl = state.fib_levels['1.0']

                    min_pip_dist = 2  # حداقل 2 پیپ واقعی
                    pip_size = _pip_size_for(MT5_CONFIG['symbol'])
                    min_abs_dist = max(min_pip_dist * pip_size, min_dist)

                    # گارد جهت - fib 1.0 همیشه باید زیر entry باشد
                    if candidate_sl >= buy_entry_price:
                        log("🚫 Skip BUY: fib 1.0 is above entry price", color='red')
                        state.reset()
                        reset_state_and_window()
                        continue
                    # اطمینان از فاصله
                    if (buy_entry_price - candidate_sl) < min_abs_dist:
                        # اگر فاصله خیلی کم است، یا SL را جابه‌جا کن یا معامله را لغو کن
                        adj = buy_entry_price - min_abs_dist
                        if adj <= 0:
                            log("🚫 Skip BUY: invalid SL distance", color='red')
                            state.reset()
                            reset_state_and_window()
                            continue
                        candidate_sl = float(adj)

                    stop = float(candidate_sl)
                    if stop >= buy_entry_price:
                        log("🚫 Skip BUY: SL still >= entry after adjust", color='red')
                        state.reset()
                        reset_state_and_window()
                        continue

                    stop_distance = abs(buy_entry_price - stop)
                    reward_end = buy_entry_price + (stop_distance * win_ratio)
                    log(f'Original stop = {stop}', color='green')
                    log(f'Original reward_end = {reward_end}', color='green')

                    # === اعمال فیلتر M15 (استراتژی S2) ===
                    m15_action, m15_reason, final_sl, final_tp, final_direction, m15_info = apply_m15_filter(
                        signal_direction='buy',
                        entry_price=buy_entry_price,
                        original_sl=stop,
                        win_ratio=win_ratio,
                        symbol=MT5_CONFIG['symbol']
                    )
                    
                    log(f'📊 M15 Filter Result: {m15_action} - {m15_reason}', color='cyan')
                    
                    # بررسی نتیجه فیلتر
                    if m15_action == 'REJECT':
                        log(f'❌ Signal REJECTED by M15 filter: {m15_reason}', color='red')
                        # ارسال ایمیل اطلاع‌رسانی رد سیگنال
                        try:
                            m15_email_info = format_m15_email_info(m15_action, m15_reason, m15_info, 'buy', '')
                            send_trade_email_async(
                                subject=f"SIGNAL REJECTED - BUY {MT5_CONFIG['symbol']}",
                                body=(
                                    f"It's for 5000 dollars account.\n\n"
                                    f"❌ TRADING SIGNAL REJECTED BY M15 FILTER ❌\n\n"
                                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                    f"Symbol: {MT5_CONFIG['symbol']}\n"
                                    f"Original Signal: BUY (Bullish Swing)\n"
                                    f"Entry Would Be: {buy_entry_price:.5f}\n"
                                    f"SL Would Be: {stop:.5f}\n"
                                    f"TP Would Be: {reward_end:.5f}\n"
                                    f"{m15_email_info}"
                                )
                            )
                        except Exception as _e:
                            log(f'Reject email failed: {_e}', color='red')
                        state.reset()
                        reset_state_and_window()
                        continue
                    
                    # تنظیم پارامترهای نهایی بر اساس نتیجه فیلتر
                    if m15_action == 'EXECUTE_REVERSED':
                        log(f'🔄 Executing REVERSED position: {final_direction.upper()}', color='blue')
                        trade_type = final_direction
                        trade_sl = final_sl
                        trade_tp = final_tp
                        trade_comment = f"REVERSED Swing (M15 {m15_info.get('direction', '')} {m15_info.get('body_ratio', 0):.0f}%)"
                    else:
                        log(f'✅ Executing ORIGINAL position: BUY', color='green')
                        trade_type = 'buy'
                        trade_sl = stop
                        trade_tp = reward_end
                        trade_comment = f"Bullish Swing {last_swing_type}"
                    
                    log(f'Final trade: {trade_type.upper()} | SL={trade_sl:.5f} | TP={trade_tp:.5f}', color='cyan')
                    
                    # لاگ سیگنال با اطلاعات نهایی (بعد از M15 filter)
                    try:
                        log_signal(
                            symbol=MT5_CONFIG['symbol'],
                            strategy="swing_fib_v1_m15_s2",
                            direction=trade_type,  # جهت نهایی (buy یا sell)
                            rr=win_ratio,
                            entry=buy_entry_price,
                            sl=trade_sl,  # SL نهایی
                            tp=trade_tp,  # TP نهایی
                            fib=state.fib_levels,
                            confidence=m15_info.get('body_ratio', None) if m15_info else None,
                            features_json=None,
                            note=f"original_signal:buy|m15_action:{m15_action}|m15_dir:{m15_info.get('direction', 'N/A') if m15_info else 'N/A'}|final_dir:{trade_type}"
                        )
                    except Exception as e:
                        log(f'log_signal failed: {e}', color='yellow')
                    
                    # گرفتن tick جدید قبل از ارسال سفارش (برای اطمینان از قیمت‌های به‌روز)
                    last_tick = mt5.symbol_info_tick(MT5_CONFIG['symbol'])
                    
                    # اگر پوزیشن ریورس شده، SL و TP رو بر اساس قیمت واقعی معامله محاسبه کن
                    if m15_action == 'EXECUTE_REVERSED' and trade_type == 'sell':
                        # برای SELL از bid استفاده میشه
                        actual_entry = last_tick.bid
                        # فاصله SL اصلی رو نگه دار
                        original_stop_distance = abs(buy_entry_price - stop)
                        trade_sl = actual_entry + original_stop_distance  # SL بالای entry برای SELL
                        trade_tp = actual_entry - (original_stop_distance * win_ratio)  # TP پایین entry
                        log(f'🔄 Recalculated for SELL: entry={actual_entry:.5f} SL={trade_sl:.5f} TP={trade_tp:.5f}', color='yellow')
                    
                    # ✨ اعمال ML Position Management (فقط برای Reversed)
                    is_reversed = (m15_action == 'EXECUTE_REVERSED')
                    actual_entry = last_tick.ask if trade_type == 'buy' else last_tick.bid
                    
                    ml_result = ml_manager.adjust_sl_tp(
                        entry=actual_entry,
                        sl=trade_sl,
                        tp=trade_tp,
                        trade_type=trade_type,
                        is_reversed=is_reversed
                    )
                    
                    if ml_result['action'] != 'NO_CHANGE':
                        log(f'🤖 ML Adjustment: {ml_result["action"]}', color='magenta')
                        log(f'   Reason: {ml_result["reason"]}', color='magenta')
                        log(f'   Original SL/TP: {trade_sl:.5f} / {trade_tp:.5f}', color='yellow')
                        trade_sl = ml_result['new_sl']
                        trade_tp = ml_result['new_tp']
                        log(f'   New SL/TP: {trade_sl:.5f} / {trade_tp:.5f}', color='green')
                        if 'win_probability' in ml_result:
                            log(f'   Win Probability: {ml_result["win_probability"]:.1%}', color='cyan')
                    
                    # ارسال سفارش با پارامترهای نهایی
                    if trade_type == 'buy':
                        result = mt5_conn.open_buy_position(
                            tick=last_tick,
                            sl=trade_sl,
                            tp=trade_tp,
                            comment=trade_comment,
                            risk_pct=MT5_CONFIG['risk_percent']
                        )
                    else:
                        result = mt5_conn.open_sell_position(
                            tick=last_tick,
                            sl=trade_sl,
                            tp=trade_tp,
                            comment=trade_comment,
                            risk_pct=MT5_CONFIG['risk_percent']
                        )
                    
                    # ارسال ایمیل غیرمسدودکننده
                    try:
                        m15_email_info = format_m15_email_info(m15_action, m15_reason, m15_info, 'buy', trade_type)
                        send_trade_email_async(
                            subject=f"NEW {trade_type.upper()} ORDER {MT5_CONFIG['symbol']} {'(REVERSED)' if m15_action == 'EXECUTE_REVERSED' else ''}",
                            body=(
                                f"It's for 5000 dollars account.\n\n"
                                f"Time: {datetime.now()}\n"
                                f"Symbol: {MT5_CONFIG['symbol']}\n"
                                f"Type: {trade_type.upper()} {'(REVERSED from BUY)' if m15_action == 'EXECUTE_REVERSED' else '(Bullish Swing)'}\n"
                                f"Entry: {last_tick.ask if trade_type == 'buy' else last_tick.bid}\n"
                                f"SL: {trade_sl}\n"
                                f"TP: {trade_tp}\n"
                                f"{m15_email_info}"
                            )
                        )
                    except Exception as _e:
                        log(f'Email dispatch failed: {_e}', color='red')

                    if result and getattr(result, 'retcode', None) == 10009:
                        log(f'✅ {trade_type.upper()} order executed successfully', color='green')
                        log(f'📊 Ticket={result.order} Price={result.price} Volume={result.volume}', color='cyan')
                        # # ارسال ایمیل غیرمسدودکننده
                        # try:
                        #     send_trade_email_async(
                        #         subject = f"Last order result",
                        #         body=(
                        #             f"Ticket={result.order}\n"
                        #             f"Price={result.price}\n"
                        #             f"Volume={result.volume}\n"
                        #         )
                        #     )
                        # except Exception as _e:
                        #     log(f'Email dispatch failed: {_e}', color='red')
                    else:
                        if result:
                            log(f'❌ {trade_type.upper()} failed retcode={result.retcode} comment={result.comment}', color='red')
                        else:
                            log(f'❌ {trade_type.upper()} failed (no result object)', color='red')
                    state.reset()

                    reset_state_and_window()
                    legs = []

                # بخش معاملات - sell statement (مطابق منطق main_saver_copy2.py)
                if last_swing_type == 'bearish' and state.second_touch:
                    # بررسی پوزیشن‌های باز قبل از ایجاد سیگنال جدید (اگر فعال باشد)
                    if TRADING_CONFIG.get('prevent_multiple_positions', True):
                        check_mode = TRADING_CONFIG.get('position_check_mode', 'all')
                        should_skip = False
                        skip_reason = ""
                        
                        if check_mode == 'all' and has_open_positions():
                            log(f"🚫 Skip SELL signal: Position(s) already open (mode: all positions)", color='yellow')
                            should_skip = True
                            skip_reason = f"Position(s) already open (mode: {check_mode})"
                        elif check_mode == 'conflicting' and has_conflicting_positions('sell'):
                            log(f"🚫 Skip SELL signal: Conflicting BUY position(s) detected", color='yellow')
                            should_skip = True
                            skip_reason = "Conflicting BUY position(s) detected"
                        
                        if should_skip:
                            log_open_positions()
                            
                            # ارسال ایمیل اطلاع‌رسانی skip شدن سیگنال SELL
                            try:
                                positions_summary = get_positions_summary()
                                send_trade_email_async(
                                    subject=f"SIGNAL SKIPPED - SELL {MT5_CONFIG['symbol']}",
                                    body=(
                                        f"It's for 5000 dollars account.\n\n"
                                        f"🚫 TRADING SIGNAL SKIPPED 🚫\n\n"
                                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                        f"Symbol: {MT5_CONFIG['symbol']}\n"
                                        f"Signal Type: SELL (Bearish Swing)\n"
                                        f"Action: SKIPPED\n"
                                        f"Reason: {skip_reason}\n"
                                        f"Check Mode: {check_mode}\n\n"
                                        f"📊 Signal Details:\n"
                                        f"Entry Price Would Be: {sell_entry_price:.5f}\n"
                                        f"Stop Loss Would Be: {state.fib_levels.get('1.0', 'N/A'):.5f}\n\n"
                                        f"📉 Fibonacci Levels:\n"
                                        f"   fib 0.0 (support): {state.fib_levels.get('0.0', 'N/A'):.5f}\n"
                                        f"   fib 0.705 (entry zone): {state.fib_levels.get('0.705', 'N/A'):.5f}\n"
                                        f"   fib 1.0 (resistance/SL): {state.fib_levels.get('1.0', 'N/A'):.5f}\n\n"
                                        f"🔒 Current Open Positions:\n{positions_summary}\n"
                                    )
                                )
                                log(f"📧 Skip signal email sent for SELL signal", color='cyan')
                            except Exception as _e:
                                log(f'Skip signal email failed: {_e}', color='red')
                            
                            state.reset()
                            reset_state_and_window()
                            continue
                    
                    log(f"📉 Sell signal triggered", color='red')
                    last_tick = mt5.symbol_info_tick(MT5_CONFIG['symbol'])
                    sell_entry_price = last_tick.bid
                    log(f'Start short position income {cache_data.iloc[-1].name}', color='red')
                    log(f'current_open_point (market bid): {sell_entry_price}', color='red')
                    # ENTRY CONTEXT (SELL): fib snapshot + touches
                    try:
                        fib = state.fib_levels or {}
                        fib0_p = fib.get('0.0')
                        fib1_p = fib.get('1.0')
                        log(
                            f"ENTRY_CTX_SELL | fib0_time={state.fib0_time} value={fib0_p} | fib705={fib.get('0.705')} | fib09={fib.get('0.9')} | fib1_time={state.fib1_time} value={fib1_p}",
                            color='cyan'
                        )
                    except Exception:
                        pass

                    pip_size = _pip_size_for(MT5_CONFIG['symbol'])
                    two_pips = 2.0 * pip_size
                    min_dist = _min_stop_distance(MT5_CONFIG['symbol'])

                    # همیشه از fib 1.0 استفاده می‌کنیم
                    candidate_sl = state.fib_levels['1.0']

                    min_pip_dist = 2.0
                    pip_size = _pip_size_for(MT5_CONFIG['symbol'])
                    min_abs_dist = max(min_pip_dist * pip_size, min_dist)

                    # گارد جهت - fib 1.0 همیشه باید بالای entry باشد
                    if candidate_sl <= sell_entry_price:
                        log("🚫 Skip SELL: fib 1.0 is below entry price", color='red')
                        state.reset()
                        reset_state_and_window()
                        continue
                    if (candidate_sl - sell_entry_price) < min_abs_dist:
                        adj = sell_entry_price + min_abs_dist
                        candidate_sl = float(adj)

                    stop = float(candidate_sl)
                    if stop <= sell_entry_price:
                        log("🚫 Skip SELL: SL still <= entry after adjust", color='red')
                        state.reset()
                        reset_state_and_window()
                        continue

                    stop_distance = abs(sell_entry_price - stop)
                    reward_end = sell_entry_price - (stop_distance * win_ratio)
                    log(f'Original stop = {stop}', color='red')
                    log(f'Original reward_end = {reward_end}', color='red')

                    # === اعمال فیلتر M15 (استراتژی S2) ===
                    m15_action, m15_reason, final_sl, final_tp, final_direction, m15_info = apply_m15_filter(
                        signal_direction='sell',
                        entry_price=sell_entry_price,
                        original_sl=stop,
                        win_ratio=win_ratio,
                        symbol=MT5_CONFIG['symbol']
                    )
                    
                    log(f'📊 M15 Filter Result: {m15_action} - {m15_reason}', color='cyan')
                    
                    # بررسی نتیجه فیلتر
                    if m15_action == 'REJECT':
                        log(f'❌ Signal REJECTED by M15 filter: {m15_reason}', color='red')
                        # ارسال ایمیل اطلاع‌رسانی رد سیگنال
                        try:
                            m15_email_info = format_m15_email_info(m15_action, m15_reason, m15_info, 'sell', '')
                            send_trade_email_async(
                                subject=f"SIGNAL REJECTED - SELL {MT5_CONFIG['symbol']}",
                                body=(
                                    f"It's for 5000 dollars account.\n\n"
                                    f"❌ TRADING SIGNAL REJECTED BY M15 FILTER ❌\n\n"
                                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                    f"Symbol: {MT5_CONFIG['symbol']}\n"
                                    f"Original Signal: SELL (Bearish Swing)\n"
                                    f"Entry Would Be: {sell_entry_price:.5f}\n"
                                    f"SL Would Be: {stop:.5f}\n"
                                    f"TP Would Be: {reward_end:.5f}\n"
                                    f"{m15_email_info}"
                                )
                            )
                        except Exception as _e:
                            log(f'Reject email failed: {_e}', color='red')
                        state.reset()
                        reset_state_and_window()
                        continue
                    
                    # تنظیم پارامترهای نهایی بر اساس نتیجه فیلتر
                    if m15_action == 'EXECUTE_REVERSED':
                        log(f'🔄 Executing REVERSED position: {final_direction.upper()}', color='blue')
                        trade_type = final_direction
                        trade_sl = final_sl
                        trade_tp = final_tp
                        trade_comment = f"REVERSED Swing (M15 {m15_info.get('direction', '')} {m15_info.get('body_ratio', 0):.0f}%)"
                    else:
                        log(f'✅ Executing ORIGINAL position: SELL', color='green')
                        trade_type = 'sell'
                        trade_sl = stop
                        trade_tp = reward_end
                        trade_comment = f"Bearish Swing {last_swing_type}"
                    
                    log(f'Final trade: {trade_type.upper()} | SL={trade_sl:.5f} | TP={trade_tp:.5f}', color='cyan')
                    
                    # لاگ سیگنال با اطلاعات نهایی (بعد از M15 filter)
                    try:
                        log_signal(
                            symbol=MT5_CONFIG['symbol'],
                            strategy="swing_fib_v1_m15_s2",
                            direction=trade_type,  # جهت نهایی (buy یا sell)
                            rr=win_ratio,
                            entry=sell_entry_price,
                            sl=trade_sl,  # SL نهایی
                            tp=trade_tp,  # TP نهایی
                            fib=state.fib_levels,
                            confidence=m15_info.get('body_ratio', None) if m15_info else None,
                            features_json=None,
                            note=f"original_signal:sell|m15_action:{m15_action}|m15_dir:{m15_info.get('direction', 'N/A')}|final_dir:{trade_type}"
                        )
                    except Exception as e:
                        log(f'log_signal failed: {e}', color='yellow')
                    
                    # گرفتن tick جدید قبل از ارسال سفارش (برای اطمینان از قیمت‌های به‌روز)
                    last_tick = mt5.symbol_info_tick(MT5_CONFIG['symbol'])
                    
                    # اگر پوزیشن ریورس شده، SL و TP رو بر اساس قیمت واقعی معامله محاسبه کن
                    if m15_action == 'EXECUTE_REVERSED' and trade_type == 'buy':
                        # برای BUY از ask استفاده میشه
                        actual_entry = last_tick.ask
                        # فاصله SL اصلی رو نگه دار
                        original_stop_distance = abs(sell_entry_price - stop)
                        trade_sl = actual_entry - original_stop_distance  # SL پایین entry برای BUY
                        trade_tp = actual_entry + (original_stop_distance * win_ratio)  # TP بالای entry
                        log(f'🔄 Recalculated for BUY: entry={actual_entry:.5f} SL={trade_sl:.5f} TP={trade_tp:.5f}', color='yellow')
                    
                    # ✨ اعمال ML Position Management (فقط برای Reversed)
                    is_reversed = (m15_action == 'EXECUTE_REVERSED')
                    actual_entry = last_tick.ask if trade_type == 'buy' else last_tick.bid
                    
                    ml_result = ml_manager.adjust_sl_tp(
                        entry=actual_entry,
                        sl=trade_sl,
                        tp=trade_tp,
                        trade_type=trade_type,
                        is_reversed=is_reversed
                    )
                    
                    if ml_result['action'] != 'NO_CHANGE':
                        log(f'🤖 ML Adjustment: {ml_result["action"]}', color='magenta')
                        log(f'   Reason: {ml_result["reason"]}', color='magenta')
                        log(f'   Original SL/TP: {trade_sl:.5f} / {trade_tp:.5f}', color='yellow')
                        trade_sl = ml_result['new_sl']
                        trade_tp = ml_result['new_tp']
                        log(f'   New SL/TP: {trade_sl:.5f} / {trade_tp:.5f}', color='green')
                        if 'win_probability' in ml_result:
                            log(f'   Win Probability: {ml_result["win_probability"]:.1%}', color='cyan')
                    
                    # ارسال سفارش با پارامترهای نهایی
                    if trade_type == 'sell':
                        result = mt5_conn.open_sell_position(
                            tick=last_tick,
                            sl=trade_sl,
                            tp=trade_tp,
                            comment=trade_comment,
                            risk_pct=MT5_CONFIG['risk_percent']
                        )
                    else:
                        result = mt5_conn.open_buy_position(
                            tick=last_tick,
                            sl=trade_sl,
                            tp=trade_tp,
                            comment=trade_comment,
                            risk_pct=MT5_CONFIG['risk_percent']
                        )
                    
                    # ارسال ایمیل غیرمسدودکننده
                    try:
                        m15_email_info = format_m15_email_info(m15_action, m15_reason, m15_info, 'sell', trade_type)
                        send_trade_email_async(
                            subject=f"NEW {trade_type.upper()} ORDER {MT5_CONFIG['symbol']} {'(REVERSED)' if m15_action == 'EXECUTE_REVERSED' else ''}",
                            body=(
                                f"It's for 5000 dollars account.\n\n"
                                f"Time: {datetime.now()}\n"
                                f"Symbol: {MT5_CONFIG['symbol']}\n"
                                f"Type: {trade_type.upper()} {'(REVERSED from SELL)' if m15_action == 'EXECUTE_REVERSED' else '(Bearish Swing)'}\n"
                                f"Entry: {last_tick.ask if trade_type == 'buy' else last_tick.bid}\n"
                                f"SL: {trade_sl}\n"
                                f"TP: {trade_tp}\n"
                                f"{m15_email_info}"
                            )
                        )
                    except Exception as _e:
                        log(f'Email dispatch failed: {_e}', color='red')
                    
                    if result and getattr(result, 'retcode', None) == 10009:
                        log(f'✅ {trade_type.upper()} order executed successfully', color='green')
                        log(f'📊 Ticket={result.order} Price={result.price} Volume={result.volume}', color='cyan')
                        # ارسال ایمیل غیرمسدودکننده
                        # try:
                        #     send_trade_email_async(
                        #         subject = f"Last order result",
                        #         body=(
                        #             f"Ticket={result.order}\n"
                        #             f"Price={result.price}\n"
                        #             f"Volume={result.volume}\n"
                        #         )
                        #     )
                        # except Exception as _e:
                        #     log(f'Email dispatch failed: {_e}', color='red')
                    else:
                        if result:
                            log(f'❌ {trade_type.upper()} failed retcode={result.retcode} comment={result.comment}', color='red')
                        else:
                            log(f'❌ {trade_type.upper()} failed (no result object)', color='red')
                    state.reset()

                    reset_state_and_window()
                    legs = []
                
                # log(f'cache_data.iloc[-1].name: {cache_data.iloc[-1].name}', color='lightblue_ex')
                # log(f'Total cache_data len: {len(cache_data)} | window_size: {window_size}', color='cyan')
                log(f'len(legs): {len(legs)} | start_index: {start_index} | {cache_data.iloc[start_index].name}', color='lightred_ex')
                log(f' ' * 80)
                log(f'-'* 80)
                log(f' ' * 80)

                # ذخیره آخرین زمان داده
                # last_data_time = cache_data.index[-1]  # این خط حذف شد چون بالا انجام شد

            # بررسی وضعیت پوزیشن‌های باز
            positions = mt5_conn.get_positions()
            if positions is None or len(positions) == 0:
                if position_open:
                    log("🏁 All positions closed", color='yellow')
                    position_open = False
            else:
                if not position_open:
                    log("🔓 Position(s) detected as open", color='cyan')
                    log_open_positions()
                    position_open = True

            manage_open_positions()

            sleep(0.5)  # مطابق main_saver_copy2.py

        except KeyboardInterrupt:
            log("🛑 Bot stopped by user", color='yellow')
            mt5_conn.close_all_positions()
            break
        except Exception as e:
            log(f' ' * 80)
            log(f"❌ Error: {e}", color='red')
            sleep(5)

    mt5_conn.shutdown()
    print("🔌 MT5 connection closed")

def _pip_size_for(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if not info:
        return 0.0001
    # برای 5/3 رقمی: 1 pip = 10 * point
    return info.point * (10.0 if info.digits in (3, 5) else 1.0)

def _min_stop_distance(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if not info:
        return 0.0003
    point = info.point
    # حداقل فاصله مجاز بروکر (stops_level) یا 3 پوینت به‌عنوان فfallback
    return max((getattr(info, 'trade_stops_level', 0) or 0) * point, 3 * point)

if __name__ == "__main__":
    main()
