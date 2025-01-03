from dataclasses import dataclass
from fileinput import close
from time import sleep
import eabot as ea
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas_ta as pta
from scipy.optimize import curve_fit
from sklearn.metrics import root_mean_squared_error
from League_of_Strategies.walk_forward_analysis import generate_walk_forward_sets, calculate_num_periods
from tqdm import tqdm
from League_of_Strategies.entry_signal import supertrend_x2_for_BTC
from League_of_Optimizer.PSO import Particle

# Import the library
from dataclasses import dataclass
from pprint import pprint
import eabot as ea
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt
import time
import csv
import threading
from dataclasses import dataclass, field

# KZM

@dataclass
class Context:
    """
    store your context here for your bot logic
    """
    first_run: bool = True
    shoot_list: list = field(default_factory=list)
    buy_counter: int = 0
    sell_counter: int = 0


    # always True parameter
    margin_fix: float = 0.0
    get_initial_balance: bool = True
    balance: float = 0.0

    # 0.2 = 20%
    pct_open: float = 0.1


def on_init_data(algo: ea.Algo) -> pd.DataFrame:
    """This function will be called once before `on_interval` function when backtest,
    and every time before `on_interval` function when live trading."""

    ######## Fix Here ##########
    print("on_init : ", floor, ceiling, spread_percentage, initial_money)

    spread = spread_percentage / 100
    # Create the new dataframe from the following data
    GAP = spread * ceiling
    amount = round((ceiling - floor) / GAP) - 1
    # the money per layer
    money_per_layer = initial_money / amount

    # Step 4 : (on Tick) Create the Trading Sheet that will use to Trade
    layer = []
    price_to_trade = []

    for i in range(amount):
        layer.append(i + 1)
        price_to_trade.append(ceiling - (GAP * (i + 1)))

    # Create a DataFrame
    df = pd.DataFrame({
        'layer': layer,
        'price_to_trade': price_to_trade,
    })

    data_df = algo.get_klines("BTC/USDT:USDT", '5m')
    data_df["highest_layer"] = df.loc[0, 'price_to_trade']
    data_df["lowest_layer"] = df.loc[amount - 1, 'price_to_trade']
    data_df['GAP'] = GAP
    data_df['money_per_layer'] = money_per_layer

    # print("Calcualte entry signal time taken:", elapsed_time, "seconds")
    # data_df.to_csv("on_init_data_div_original.csv",index=False)
    return data_df[["lowest_layer", "highest_layer", "GAP", "money_per_layer"]]


def on_interval(algo: ea.Algo, ctx: Context):
    """
    this function will be called every interval
    """
    if ctx.get_initial_balance:
        # do at the First RUN
        ctx.balance = algo.get_balance().available
        ctx.margin_fix = get_margin_fix(balance=ctx.balance, pct_open=ctx.pct_open)
        algo.log_info(f"balance = {ctx.balance} | pct_open = {ctx.pct_open}")
        ctx.get_initial_balance = False

    # NESSESSARY DATA DEFINE
    symbol = "BTC/USDT:USDT"
    close = algo.get_last_price(symbol)
    position = algo.get_positions()
    position_long = position[symbol]["LONG"]
    position_short = position[symbol]["SHORT"]
    status = get_status(position_long, position_short)
    balance = algo.get_balance().available

    lowest_layer = algo.get_data('lowest_layer')
    highest_layer = algo.get_data('highest_layer')
    GAP = algo.get_data('GAP')
    money_per_layer = algo.get_data('money_per_layer')

    algo.log_info(f"close = {close}| long_q = {position_long.quantity}| short_q = {position_short.quantity} ")
    if ctx.first_run:
        quantity = money_per_layer/close
        algo.buy(quantity=quantity, symbol=symbol, position_side="LONG")

        shoot_buy_list = []
        shoot_sell_list = []
        # Setup the layer
        _ = close
        while _ >= lowest_layer:
            _ -= GAP
            shoot_buy_list.insert(0, _)

        _ = close
        while _ <= highest_layer:
            _ += GAP
            shoot_sell_list.append(_)

        # shoot_buy_list = shoot_buy_list.sort(reverse=True)
        ctx.shoot_list = shoot_buy_list + [close] + shoot_sell_list

        ctx.buy_counter = ctx.shoot_list.index(close) - 1
        ctx.sell_counter = ctx.shoot_list.index(close) + 1
        ctx.active = ctx.shoot_list.index(close)

        # finish the first run
        ctx.first_run = False

    if not ctx.first_run:
        try:
            if close <= ctx.shoot_list[ctx.buy_counter]:
                # act 1 : buy the order
                quantity = money_per_layer / close
                algo.buy(quantity=quantity, symbol=symbol, position_side="LONG")
                # act 2 : update the sell
                ctx.buy_counter -= 1
                ctx.sell_counter -= 1
        except:
            print('The value is the same')

        if position_long.quantity > 0:
            try:
                if close >= ctx.shoot_list[ctx.sell_counter]:
                    # act 1 : buy the order
                    quantity = money_per_layer / close

                    if position_long.quantity < quantity:
                        quantity = position_long.quantity

                    algo.sell(quantity=quantity, symbol=symbol, position_side="LONG")
                    # act 2 : update the buy
                    ctx.buy_counter += 1
                    ctx.sell_counter += 1
            except:
                print('The value is the same')

        if position_long.quantity == 0:
            try:
                if close >= ctx.shoot_list[ctx.sell_counter]:
                    # act 1 : buy the order
                    quantity = money_per_layer / close

                    algo.buy(quantity=quantity, symbol=symbol, position_side="LONG")
                    # act 2 : update the buy
                    ctx.buy_counter += 1
                    ctx.sell_counter += 1
            except:
                print('The value is the same')



    return

def get_tp_price_open_long(pct_tp: float, entry_price: float) -> float:
    return (1 + pct_tp) * entry_price


def get_tp_price_open_short(pct_tp: float, entry_price: float) -> float:
    return (1 - pct_tp) * entry_price


def sl_price_long(pct_sl: float, entry_price: float) -> float:
    return entry_price * (1 - pct_sl)


def sl_price_short(pct_sl: float, entry_price: float) -> float:
    return entry_price * (1 + pct_sl)


def get_RSI(close_sr: pd.Series, window: int) -> pd.Series:
    rsi = RSIIndicator(close_sr, window=window).rsi()
    return rsi

def stop_loss_long(
    pct_sl: float, position_long: ea.Position, close: float
) -> bool:
    return close < position_long.entry_price * (1 - pct_sl)


def stop_loss_short(
    pct_sl: float, position_short: ea.Position, close: float
) -> bool:
    return close > position_short.entry_price * (1 + pct_sl)


def is_less_than_lift_stop(lift_stop_after: int, count_cut: int) -> bool:
    return lift_stop_after > count_cut


def lift_long(pct_lift: float, position_long: ea.Position, close: float) -> bool:
    return close > position_long.entry_price * (1 + pct_lift)


def lift_short(
    pct_lift: float, position_short: ea.Position, close_sr: float
) -> bool:
    return close_sr < position_short.entry_price * (1 - pct_lift)


"""""" """
Accounting part
""" """"""


def get_margin_fix(balance: float, pct_open: float) -> float:
    return balance * pct_open


def get_quantity(margin_fix: float, close:float, position: ea.Position) -> float:
    return margin_fix * position.leverage / close


def martingale_margin(margin_fix: float, count_lost: int) -> float:
    return margin_fix * pow(2, count_lost)


def get_status(long: ea.Position, short: ea.Position):
    position_long = long.quantity
    position_short = short.quantity
    if position_long > 0 and position_short == 0:
        return "LONG"
    elif position_long == 0 and position_short > 0:
        return "SHORT"
    elif position_long == 0 and position_short == 0:
        return "OPEN"

"""
Statistic Data
"""
def get_stats(trade_df:pd.DataFrame,summary_df: pd.DataFrame,trades_bt:dict) -> pd.DataFrame:
    df_summary = summary_df
    df_trade = trade_df
    # trade_bt df calculation
    total_win = trades_bt['won']['total']
    total_lost = trades_bt['lost']['total']
    average_position = trades_bt['len']['average']
    max_position = trades_bt['len']['max']
    min_position = trades_bt['len']['min']
    long_total = trades_bt['long']['total']
    short_total = trades_bt['short']['total']
    if long_total != 0:
        long_winrate=trades_bt['long']['won']/long_total
    else:
        long_winrate = 0
    if short_total != 0:
        short_winrate=trades_bt['short']['won']/short_total
    else:
        short_winrate = 0
    winrate = round(total_win/trades_bt['total']['closed'],2)*100
    total_trade = trades_bt['total']['closed']
    average_win = round(trades_bt['won']['pnl']['average'],2)
    average_loss = round(trades_bt['lost']['pnl']['average'],2)
    max_win = trades_bt['won']['pnl']['max']
    max_loss = trades_bt['lost']['pnl']['max']
    # trade_df df calculation
    df_trade["volume"] = df_trade["quantity"] * df_trade["price"]
    df_trade["cummulative_volume"] = df_trade["volume"].cumsum()
    trade_df['time'] = pd.to_datetime(trade_df['time'])
    trade_df.set_index('time', inplace=True)
    monthly_volume = df_trade['volume'].resample('M').sum()
    average_monthly_volume = round(monthly_volume.mean()/summary_df["port_value"].iloc[0],2)
    maximum_monthly_volume = round(monthly_volume.max()/summary_df["port_value"].iloc[0],2)
    minimum_monthly_volume = round(monthly_volume.min()/summary_df["port_value"].iloc[0],2)
    # df_summary df calculation
    df_summary["max_NAV"] = df_summary["port_value"].cummax()
    df_summary["drawdown"] = (1 - df_summary["port_value"] / df_summary["max_NAV"]) * 100
    df_summary["maximum_drawdown"] = df_summary["drawdown"].cummax()
    maximum_drawdown = round(df_summary["maximum_drawdown"].max(), 2)
    # df_summary["recovery_period_1"] = np.where(df_summary["port_value"] < df_summary["max_NAV"], 1, 0)
    # grouper3 = (df_summary.recovery_period_1 != df_summary.recovery_period_1.shift()).cumsum()
    # df_summary["recovery_period"] = df_summary.groupby(grouper3).cumsum()["recovery_period_1"]
    # max_recovery_period = int(df_summary["recovery_period"].max())
    df_summary["absolute_drawdown"] = np.where(
        df_summary["port_value"] < df_summary["max_NAV"], df_summary["max_NAV"] - df_summary["port_value"], 0
    )
    max_absolute_drawdown = round(int(df_summary["absolute_drawdown"].max())*100/df_summary["port_value"].iloc[0],2)
    df_summary['port_value_pct_change'] = (df_summary['port_value']/df_summary['port_value'].shift(1))-1
    monthly_data = df_summary.resample('M', on='time').agg({'port_value': ['first', 'last']})
    positive_months = monthly_data[monthly_data['port_value', 'last'] >= monthly_data['port_value', 'first']]
    num_positive_months = len(positive_months)
    monthly_data = df_summary.resample('M', on='time').count()
    total_months = len(monthly_data)
    pct_positive_months = round(num_positive_months*100/total_months,0)
    weekly_data = df_summary.resample('W', on='time').agg({'port_value': ['first', 'last']})
    positive_weeks = weekly_data[weekly_data['port_value', 'last'] >= weekly_data['port_value', 'first']]
    num_positive_weeks = len(positive_weeks)
    weekly_data = df_summary.resample('W', on='time').count()
    total_weeks = len(weekly_data)
    pct_positive_week = round(num_positive_weeks*100/total_weeks,0)
    apr = round(
        (df_summary["port_value"].iloc[-1] - df_summary["port_value"].iloc[0])
        * 100
        / df_summary["port_value"].iloc[0],
        2,
    )
    trade_volme = df_trade["cummulative_volume"].iloc[-1]
    sharpe_ratio = df_summary['port_value_pct_change'].mean()/df_summary['port_value_pct_change'].std()
    prom = round((average_win*(total_win-np.sqrt(total_win))+(average_loss*(total_lost+np.sqrt(total_lost))))*100/df_summary["port_value"].iloc[0],2)
    RoMAD = round(apr/max_absolute_drawdown,2)
    slope = (df_summary['port_value'].iloc[-1] - df_summary['port_value'].iloc[0]) / len(df_summary)
    df_summary['benchmark'] = slope * df_summary['port_value'].index + df_summary['port_value'].iloc[0]
    df_summary['offset_benchmark'] = abs(df_summary['benchmark']-df_summary['port_value'])
    offset_total = df_summary['offset_benchmark'].sum()
    print(
        f"winrate = {winrate}% | RoMaD = {RoMAD} | maximum drawdown = {maximum_drawdown}% | APR = {apr}% | average trade volume = {average_monthly_volume}X | maximum trade volume = {maximum_monthly_volume}X | minimum trade volume = {minimum_monthly_volume}X | trade count {total_trade} | max_absolute_dd = {max_absolute_drawdown}% | sharp ratio = {sharpe_ratio} | % positive months = {pct_positive_months}% | % positive weeks = {pct_positive_week}%"
    )
    statistice_data = {
        "APR": apr,
        "total_trade": total_trade,
        "total_win" : total_win,
        "total_lost" : total_lost,
        "winrate": winrate,
        "maximum_drawdown": maximum_drawdown,
        "total trade_volume": trade_volme,
        "average trade volume per month": average_monthly_volume,
        "maximum trade volume per month": maximum_monthly_volume,
        "minimum trade volume per month": minimum_monthly_volume,
        "average_win": average_win,
        "average_loss": average_loss,
        "max_win": max_win,
        "max_loss": max_loss,
        "average_position":average_position,
        "max_position":max_position,
        "min_position":min_position,
        "RoMAD": RoMAD,
        "PROM" : prom,
        # "max_recovery_period": max_recovery_period,
        "max_absolute_drawdown": max_absolute_drawdown,
        "sharpe_ratio":sharpe_ratio,
        "long_total": long_total,
        "long_winrate":long_winrate,
        "short_total": short_total,
        "short_winrate":short_winrate,
        "%_positive_month":pct_positive_months,
        "%_positive_week":pct_positive_week,
    }
    statistice_data_df = pd.DataFrame.from_dict(statistice_data, orient="index")
    df_trade = df_trade.drop(['volume','cummulative_volume'], axis=1)
    df_summary = df_summary.drop(['max_NAV','drawdown','maximum_drawdown','absolute_drawdown'], axis=1)
    trade_df = df_trade
    summary_df = df_summary
    # return statistice_data_df
    return statistice_data_df, winrate, apr,prom,total_trade,max_absolute_drawdown,RoMAD,sharpe_ratio,pct_positive_months,pct_positive_week,offset_total  #type:ignore

def read_binancekline(path):
    df = pd.read_csv(path, parse_dates=["open_time"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df.set_index("open_time")


# PSO objective function
def objective_function(params, start, end, l_interval, s_interval):
    ######## Fix Here ##########
    # config_list_long = [16, 19, 16, 4, 50, 65] # [16, 19, 16, 4, 50, 65] [16, 10, 20, 8, 30, 70]
    # config_list_short = [10, 17, 19, 4, 35, 60] #[10, 17, 19, 4, 35, 60] [11, 18, 20, 8, 25, 70]
    # # rsi_window_long:int = config_list_long[0]
    # # sma_length_long:int = config_list_long[1]
    # # pivot_far_window_long: int = config_list_long[2]
    # # pivot_near_window_long: int = config_list_long[3]
    # # rsi_thes_long: int = config_list_long[4]

    # rsi_window_short:int = config_list_short[0]
    # sma_length_short:int = config_list_short[1]
    # pivot_far_window_short: int = config_list_short[2]
    # pivot_near_window_short: int = config_list_short[3]
    # rsi_thes_short: int = config_list_short[5]

    ######## Fix Here ##########
    floor_value, ceiling_value, spread_percentage_value = params

    global TEST_PERIOD_START, TEST_PERIOD_END, INTERVAL, SMALL_INTERVAL
    global floor, ceiling, spread_percentage, initial_money
    global kline_dict

    ########## FIX HERE ###########
    floor = floor_value * 1000
    ceiling = ceiling_value * 1000
    spread_percentage = spread_percentage_value
    initial_money = 1000


    TEST_PERIOD_START = start
    TEST_PERIOD_END = end
    INTERVAL = l_interval
    SMALL_INTERVAL = s_interval

    # print("obj_func : ", LOOK_BACK, ADX_value, BB_value, BB_std_value)
    print('start', TEST_PERIOD_START,
          'end', TEST_PERIOD_END)
    
    try :
        ctx = Context()
        trade_df, summary_df, positions_df, trades_bt = ea.run_backtest(
            ctx,
            asset="USDT",
            on_interval_func=on_interval,
            large_interval=INTERVAL,
            small_interval=SMALL_INTERVAL,
            start_date=TEST_PERIOD_START,
            end_date=TEST_PERIOD_END,
            kline_dict=kline_dict,  # type:ignore
            balance_asset_initial=1_000,
            pct_fee=0.05 / 100,
            leverage=1,
            on_init_data_func=on_init_data,
            is_sl_before_tp=True,
        )

        statistice_data_df, winrate, apr,prom,total_trade,max_absolute_drawdown,RoMAD,sharpe_ratio,pct_positive_months,pct_positive_week,offset_total = get_stats(trade_df=trade_df, summary_df=summary_df,trades_bt=trades_bt)
        print(f"winrate = {winrate}% | APR = {apr}% | trade count {total_trade}")    
        score = (0.15 * apr) + (0.3 * sharpe_ratio) - (max_absolute_drawdown * 0.5)
        profit = trade_df['realized_profit'].sum()
    except:
        score = 0
        total_trade=0
        profit=0
        max_absolute_drawdown=0
    return score, total_trade, profit ,max_absolute_drawdown # Negate since we want to maximize


def optimize_strategy_with_symbol_and_tf_by_PSO(sym, tf):
    global SYMBOL, config_list_long, config_list_short, risk_percent_of_balance, default, pct_open, LEVERAGE
    global kline_dict
    global floor, ceiling, spread_percentage, initial_money

    # ----------------- Your Logic ------------------
    SYMBOL = f"{sym}/USDT:USDT"
    SYMBOL_FILE_NAME = f"OKX_{sym}-USDT-SWAP"
    LOAD_SYMBOLS = [SYMBOL]
    INTERVAL = tf
    SMALL_INTERVAL = "5m"
    LOAD_INTERVALS = [INTERVAL]
    PCT_FEE = 0.0005
    LEVERAGE = 10
    ASSET = "USDT"
    PCT_SLIPPAGE_BACKTEST = None  # for market = None and for 0.01 = 1%
    PCT_SLIPPAGE_TOLERANCE_LIVE = 0.05 / 100
    # Common parameters
    pct_open = 0.15
    count_placed_order_max: int = 4


    ######## Fix Here ##########
    # Input parameters
    floor = 10_000  # Starting amount
    ceiling = 40_000  # Maximum amount
    spread_percentage = 1  # %spread (e.g., 10 for 10%)
    initial_money = 1000


    # Optimization Setup
    param_1_name = 'floor'
    param_1_lower_bound, param_1_upper_bound = 5, 30
    param_2_name = 'ceiling'
    param_2_lower_bound, param_2_upper_bound = 40, 70
    param_3_name = 'spread_percentage'
    param_3_lower_bound, param_3_upper_bound = 1, 20
    # param_4_name = 'initial_money'
    # param_4_lower_bound, param_4_upper_bound = 1, 5

    strategy_name = "KZM_Geometric_x_PSO"
    side = "LONG"

    ### Walk Forward Setup ###
    TEST_PERIOD_START = "2023-01-01"
    TEST_PERIOD_END = "2024-05-21"
    optimization_length_months = 5
    walk_forward_length_months = 5

    ######## Fix Here ##########
    # sup_m_1, sup_m_2, RSI_Period = params
    # Particle Swarm Setup : For Train Period
    bounds = [  (param_1_lower_bound, param_1_upper_bound),
                (param_2_lower_bound, param_2_upper_bound), 
                (param_3_lower_bound, param_3_upper_bound),
                # (param_4_lower_bound, param_4_upper_bound),
                # (param_5_lower_bound, param_5_upper_bound), 
                # (param_6_lower_bound, param_6_upper_bound),
                # (param_7_lower_bound, param_7_upper_bound),
                # (param_8_lower_bound, param_8_upper_bound), 
                # (param_9_lower_bound, param_9_upper_bound),
                # (param_10_lower_bound, param_10_upper_bound),
                ]  
    nv = len(bounds)  # number of optimize variable

    mm = 1  # minimize = -1 , Maximize = 1
    # particle_size = 2  # num of particles
    # iteration = 3  # max num of iteration
    particle_size = 8  # num of particles
    iteration = 100  # max num of iteration

    w = 1  # Weight
    c1 = 2  # cognative
    c2 = 4  # social

    no_improvement_count = 0  # Counter for early stopping
    no_improvement_not_exceed = 3

    path = f"./0-Load data/binance-public-data/"
    kline_dict = {
        (f"{SYMBOL}", "4H"): read_binancekline(
            f"{path}/{SYMBOL_FILE_NAME}_4H.csv"),
        (f"{SYMBOL}", "1H"): read_binancekline(
            f"{path}/{SYMBOL_FILE_NAME}_1H.csv"),
        (f"{SYMBOL}", "30m"): read_binancekline(
            f"{path}/{SYMBOL_FILE_NAME}_30m.csv"),
        (f"{SYMBOL}", "15m"): read_binancekline(
            f"{path}/{SYMBOL_FILE_NAME}_15m.csv"),
        (f"{SYMBOL}", "5m"): read_binancekline(
            f"{path}/{SYMBOL_FILE_NAME}_5m.csv"),
    }

    num_periods = calculate_num_periods(start_year=TEST_PERIOD_START,
                                        end_year=TEST_PERIOD_END,
                                        optimization_length_months=optimization_length_months,
                                        walk_forward_length_months=walk_forward_length_months)

    walk_forward_sets = generate_walk_forward_sets(TEST_PERIOD_START,
                                                    optimization_length_months,
                                                    walk_forward_length_months,
                                                    num_periods, TEST_PERIOD_END)

    if mm == -1:
        initial_fitness = float("inf")
    if mm == 1:
        initial_fitness = -float("inf")

    for sheet_num, walk_forward in tqdm(enumerate(walk_forward_sets)):

        TEST_PERIOD_START = walk_forward['optimization_period'][0]
        TEST_PERIOD_END = walk_forward['optimization_period'][1]

        # Run the Walk Forward Analysis
        fitness_global_best_particle_position = initial_fitness
        global_best_particle_position = []
        swarm_particle = []
        all_explore_position = []
        all_evaluate_result = []

        ### FIX HERE ###
        total_trade_list = []
        profit_list = []
        max_absolute_drawdown_list = []

        for i in range(particle_size):
            swarm_particle.append(Particle(bounds = bounds, initial_fitness = initial_fitness, nv = nv,  mm = mm,
                                           w = w,  c1 = c1,  c2 = c2))

        A = []
        for i in range(iteration):
            for j in range(particle_size):
                swarm_particle[j].evaluate(objective_function, 
                                           start=TEST_PERIOD_START, end=TEST_PERIOD_END,
                                           l_interval = INTERVAL, 
                                           s_interval = SMALL_INTERVAL)
                # log all evaluation
                all_explore_position.append(swarm_particle[j].particle_position)
                all_evaluate_result.append(swarm_particle[j].fitness_particle_position)

                ## Score Equation ##
                total_trade_list.append(swarm_particle[j].total_trade)     
                profit_list.append(swarm_particle[j].profit)     
                max_absolute_drawdown_list.append(swarm_particle[j].max_absolute_drawdown)     

                if mm == -1:
                    if swarm_particle[j].fitness_particle_position < fitness_global_best_particle_position:
                        global_best_particle_position = list(
                            swarm_particle[j].particle_position)  # update position
                        fitness_global_best_particle_position = float(
                            swarm_particle[j].fitness_particle_position)
                        no_improvement_count = 0  # Reset the counter
                    else:
                        no_improvement_count += 1

                if mm == 1:
                    if swarm_particle[j].fitness_particle_position > fitness_global_best_particle_position:
                        global_best_particle_position = list(
                            swarm_particle[j].particle_position)  # update position
                        fitness_global_best_particle_position = float(
                            swarm_particle[j].fitness_particle_position)
                        no_improvement_count = 0  # Reset the counter
                    else:
                        no_improvement_count += 1

            if no_improvement_count >= no_improvement_not_exceed:
                print(f"Stopping early at iteration {i} due to no improvement in global best for 3 iterations.")
                break

            for j in range(particle_size):
                swarm_particle[j].update_velocity(global_best_particle_position)
                swarm_particle[j].update_position(bounds)

        optimization_file_path = f'./data/strategy_optimization/{strategy_name}/{side}/{sym}_{tf}_sheet_{sheet_num + 1}'
        writer = pd.ExcelWriter(f"{optimization_file_path}.xlsx", engine="xlsxwriter")
        log_df = pd.DataFrame(all_explore_position)

        ######### FIX HERE ###########
        log_df = log_df.rename(columns={0: f'{param_1_name}',
                                        1: f'{param_2_name}',
                                        2: f'{param_3_name}', 
                                        # 3: f'{param_4_name}',
                                        # 4: f'{param_5_name}',
                                        # 5: f'{param_6_name}', 
                                        # 6: f'{param_7_name}',
                                        # 7: f'{param_8_name}',
                                        # 8: f'{param_9_name}', 
                                        # 9: f'{param_10_name}',

                                        })  # Rename columns of 4H data
        
        
        log_df['total_trade'] = total_trade_list
        log_df['norm_total_trade'] = (log_df['total_trade'] - log_df['total_trade'].min()) / (log_df['total_trade'].max() - log_df['total_trade'].min())

        log_df['profit'] = profit_list
        log_df['norm_profit'] = (log_df['profit'] - log_df['profit'].min()) / (log_df['profit'].max() - log_df['profit'].min())
        
        log_df['max_absolute_drawdown'] = max_absolute_drawdown_list
        log_df['norm_max_dd'] = (log_df['max_absolute_drawdown'] - log_df['max_absolute_drawdown'].min()) / (log_df['max_absolute_drawdown'].max() - log_df['max_absolute_drawdown'].min())

        log_df['rank_score'] = all_evaluate_result
        log_df['norm_rank_score'] = log_df['norm_total_trade'] + log_df['norm_profit'] - log_df['norm_max_dd']

        log_df = log_df.sort_values(by='rank_score', ascending=False)
        log_df.to_excel(writer, sheet_name=f'sheet_{sheet_num + 1}')
        writer.close()
        ### End of Optimization ###

        ### Start walk forward ###
        TEST_PERIOD_START = walk_forward['walk_forward_period'][0]
        TEST_PERIOD_END = walk_forward['walk_forward_period'][1]

        excel_df = pd.read_excel(f"{optimization_file_path}.xlsx",
                                    f'sheet_{sheet_num + 1}')

        ######### Fix HERE #############
        rank_1_in_the_swarn = excel_df[[f'{param_1_name}', 
                                        f'{param_2_name}', 
                                        f'{param_3_name}',
                                        # f'{param_4_name}',
                                        # f'{param_5_name}', 
                                        # f'{param_6_name}',
                                        # f'{param_7_name}', 
                                        # f'{param_8_name}', 
                                        # f'{param_9_name}',
                                        # f'{param_10_name}'
                                        ]].iloc[0].tolist()

        score, total_trade, profit, max_absolute_drawdown = objective_function(rank_1_in_the_swarn,
                                                                                    start=TEST_PERIOD_START, end=TEST_PERIOD_END,
                                                                                    l_interval = INTERVAL, 
                                                                                    s_interval = SMALL_INTERVAL)
        
        ######### Fix HERE #############
        walk_log_data = {
            'SYMBOL': [SYMBOL],
            'TF': [INTERVAL],
            'START': [TEST_PERIOD_START],
            'END': [TEST_PERIOD_END],
            'Test_Name': ["walk-forward-profile"],
            f'{param_1_name}': [rank_1_in_the_swarn[0]],
            f'{param_2_name}': [rank_1_in_the_swarn[1]],
            f'{param_3_name}': [rank_1_in_the_swarn[2]],
            # f'{param_4_name}': [rank_1_in_the_swarn[3]],
            # f'{param_5_name}': [rank_1_in_the_swarn[4]],
            # f'{param_6_name}': [rank_1_in_the_swarn[5]],
            # f'{param_7_name}': [rank_1_in_the_swarn[6]],
            # f'{param_8_name}': [rank_1_in_the_swarn[7]],
            # f'{param_9_name}': [rank_1_in_the_swarn[8]],
            # f'{param_10_name}': [rank_1_in_the_swarn[9]],
            'trade_count' : [total_trade],
            'max_ab_dd' : [max_absolute_drawdown],
            'rank_score': [score],
            'profit': [profit],
        }
        walk_log_file = f'./data/strategy_optimization/{strategy_name}/{side}/walk_forward_profile.xlsx'
        try:
            df = pd.read_excel(walk_log_file)
            update_df = pd.DataFrame(walk_log_data)
            df = pd.concat([df, update_df])
            df.to_excel(f'{walk_log_file}', index=False, )
        except:
            log_df = pd.DataFrame(walk_log_data)
            log_df.to_excel(f'{walk_log_file}', index=False, )





