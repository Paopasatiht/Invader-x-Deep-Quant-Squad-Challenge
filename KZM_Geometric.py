import time
from cmath import isnan
from ctypes.wintypes import LARGE_INTEGER
from dataclasses import dataclass, field
from fileinput import close
from pprint import pprint
from time import sleep
import eabot as ea
import pandas as pd
import numpy as np
import pandas_ta as pta
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# The Libraies that use below are
# cmath :)
# ctypes :)
# dataclasses :)
# fileinput :) 
# pprint
# time
# eabot # this one is exclusivly
# pandas :)
# numpy
# pandas_ta
# typing

# Workign Log
# Add ATR to the system
# - use ATR * 2


# ----------------- Your Logic ------------------
# """
# Put the Pre-define the variable that want to use in the logic
# """
SYMBOL = "BTC/USDT:USDT"
LOAD_SYMBOLS = [SYMBOL]
INTERVAL = "5m"
SMALL_INTERVAL = "5m"
LOAD_INTERVALS = [INTERVAL]
PCT_FEE = 0.0005
LEVERAGE = 10
ASSET = "USDT"
PCT_SLIPPAGE_BACKTEST = None # for market = None and for 0.01 = 1%
PCT_SLIPPAGE_TOLERANCE_LIVE = 0.05/100
TEST_PERIOD_START = "2021-02-01"
TEST_PERIOD_END = "2024-12-31"


# Common parameters
pct_open: float = 0.15
count_placed_order_max: int = 4


@dataclass
class Context:
    """
    store your context here for your bot logic
    """
    first_run: bool = True

    shoot_list: list = field(default_factory=list)

    buy_counter: int = 0
    sell_counter: int = 0


    # 0.2 = 20%
    pct_open: float = 0.1

    # always True parameter
    margin_fix: float = 0.0
    get_initial_balance: bool = True
    balance: float = 0.0


    #### OPEN ORDER CHECK ORDER IS ACTIVE
    ### Grid Tools Box
    grid_buy_list: list = field(default_factory=list)
    grid_sell_list: list = field(default_factory=list)

    grid_buy_bullet:float = 0.0
    grid_sell_bullet:float = 0.0

    grid_quant_LONG: list = field(default_factory=list)
    grid_quant_SHORT: list = field(default_factory=list)

    grid_shoot_LONG_counter:int = 0
    grid_shoot_SHORT_counter: int = 0

    ####  TRAILING STOP ####
    first_trailing: bool = False
    second_trailing: bool = False
    first_close:bool = False

    ##### REAL TIME ANIMATION #####
    ani_index: list = field(default_factory=list)
    ani_close_price: list = field(default_factory=list)
    ani_demo: list = field(default_factory=list)

    # second axis
    ani_equity_move: list = field(default_factory=list)

    # object
    ani_open_order: list = field(default_factory=list)


def on_init_data(algo: ea.Algo) -> pd.DataFrame:
    """
    this function will be called after read csv or get data from exchange
    """

    # Step 3 : Set up the input parameter
    floor = 30_000  # Starting amount
    ceiling = 70_000  # Maximum amount
    spread_percentage = 1  # %spread (e.g., 10 for 10%)
    spread = spread_percentage / 100

    # Create the new dataframe from the following data
    GAP = spread * ceiling
    amount = round((ceiling - floor) / GAP) - 1

    # Step 5 : Set up the money to use
    initial_money = 1500
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
    data_df["lowest_layer"] = df.loc[amount-1, 'price_to_trade']
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
        quantity = money_per_layer / close
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
"""
Utility section
"""


"""""" """
Accounting part
""" """"""

def get_margin_fix(balance: float, pct_open: float) -> float:
    return balance * pct_open

def get_quantity(margin_fix: float, close:float, position: ea.Position) -> float:
    return margin_fix * position.leverage / close

def get_quantity_raw(margin_fix: float, close:float, leverage: float) -> float:
    return margin_fix * leverage / close

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
    elif position_long > 0 and position_short > 0:
        return "BOTH"

if __name__ == "__main__":

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
        monthly_data.columns = monthly_data.columns.droplevel(0)
        monthly_data['monthly_return'] = (monthly_data['last'] / monthly_data['first'] - 1) * 100
        positive_months = monthly_data[monthly_data['last'] >= monthly_data['first']]
        num_positive_months = len(positive_months)
        total_months = len(monthly_data)
        pct_positive_months = round(num_positive_months * 100 / total_months, 0)
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
        RoMAD = round(apr/max_absolute_drawdown,3)
        slope = (df_summary['port_value'].iloc[-1] - df_summary['port_value'].iloc[0]) / len(df_summary)
        df_summary['benchmark'] = slope * df_summary['port_value'].index + df_summary['port_value'].iloc[0]
        df_summary['offset_benchmark'] = df_summary['port_value']-df_summary['benchmark']
        offset_total = df_summary['offset_benchmark'].sum()
        minimum_return = round(monthly_data['monthly_return'].min(),2)
        maximum_return = round(monthly_data['monthly_return'].max(),2)
        std_dev_monthly_return = round(monthly_data['monthly_return'].std(),2)
        average_monthly_return = round(monthly_data['monthly_return'].mean(),2)
        # print('monthly data')
        # print(monthly_data)
        # pd.DataFrame(monthly_data).to_csv('monthly_data.csv')
        print(
            f"winrate = {winrate}% | RoMaD = {RoMAD} | maximum drawdown = {maximum_drawdown}% | APR = {apr}% | average trade volume = {average_monthly_volume}X | maximum trade volume = {maximum_monthly_volume}X | minimum trade volume = {minimum_monthly_volume}X | trade count {total_trade} | max_absolute_dd = {max_absolute_drawdown}% | sharpe ratio = {sharpe_ratio} | % positive months = {pct_positive_months}% | % positive weeks = {pct_positive_week} | % average monthly return = {average_monthly_return} | % maximum monthly return = {maximum_return} | % minimum monthly return = {minimum_return}% | SD % monthly return = {std_dev_monthly_return} %"
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

    kline_dict = {
    # ("BTC/USDT:USDT", "1m"): read_binancekline("./0-Load data/binance-public-data/OKX_BTC-USDT-SWAP_1m.csv"),
    ("BTC/USDT:USDT", "5m"): read_binancekline("./0-Load data/binance-public-data/OKX_BTC-USDT-SWAP_5m.csv"),
    ("BTC/USDT:USDT", "1H"): read_binancekline("./0-Load data/binance-public-data/OKX_BTC-USDT-SWAP_1H.csv"),
    ("BTC/USDT:USDT", "4H"): read_binancekline("./0-Load data/binance-public-data/OKX_BTC-USDT-SWAP_4H.csv"),
    }

    """"
    this will excecution part
    """
    ctx = Context()
    trade_df, summary_df, positions_df,trades_bt = ea.run_backtest(
        ctx,
        asset = "USDT",
        on_interval_func=on_interval,
        large_interval=INTERVAL,
        small_interval=SMALL_INTERVAL,
        start_date=TEST_PERIOD_START,
        end_date=TEST_PERIOD_END,
        kline_dict=kline_dict, #type:ignore
        balance_asset_initial=1500,
        pct_fee=0.05/100,
        leverage=1,
        on_init_data_func=on_init_data,
        is_sl_before_tp=True,
    )

    print('Finish Running')
    # import matplotlib.pyplot as plt
    # import matplotlib.gridspec as gridspec
    # import matplotlib.animation as animation
    # import numpy as np
    #
    # ### ANIMATION TIME ###
    # # x
    # # y
    # # index generate
    # x = ctx.ani_index
    # y = ctx.ani_close_price
    # y2 = ctx.ani_equity_move
    # y3 = ctx.ani_open_order
    # frame_amount = len(ctx.ani_index)
    #
    # def update_plot(num):
    #     print(num)
    #     if y3[num] != 0:
    #         open_order.set_data(x[0:num+1], y3[0:num+1])
    #
    #     close_price_trajectory.set_data(x[0:num], y[0:num])
    #     equity_trajectory.set_data(x[0:num], y2[0:num])
    #
    #     text_close_price.set_text("BTC Price : " + str(round(y[num], 2)))
    #     text_equity.set_text("Equity : " + str(int(y2[num])) + ' $')
    #
    #
    #
    #     return close_price_trajectory, equity_trajectory, text_close_price, text_equity, open_order
    #
    # fig = plt.figure(figsize=(16, 9), dpi=144, facecolor=(0.8, 0.8, 0.8, 0.8))
    # gs = gridspec.GridSpec(2,2)
    #
    # # Subplot 1
    # ax0 = fig.add_subplot(gs[0, :], facecolor=(0.9, 0.9, 0.9, 0.9)) # row 1 all column
    # ax1 = ax0.twinx()
    #
    # close_price_trajectory, = ax0.plot([], [], 'black', linewidth=2)
    # equity_trajectory, = ax1.plot([], [], 'blue', linewidth=2)
    #
    # open_order, = ax0.plot([], [], '^', color='g', markersize=12)
    #
    #
    # plt.xlim(x[0], x[-1])
    # ax0.set_ylim(min(y), max(y))
    # ax1.set_ylim(min(y2), max(y2))

    ## Text
    # text_close_price = ax0.text(0.98, 0.1, '', transform=ax0.transAxes, ha='right', va='bottom',
    #                             style='italic', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
    # text_equity = ax0.text(0.98, 0.2, '', transform=ax0.transAxes, ha='right', va='bottom',
    #                         style='italic', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
    # # Copyright
    # copyright = ax0.text(0.5, 1.2, '© Pa-satith Chantawut', transform=ax0.transAxes, ha='center', va='top')
    #
    # backtester_animation = animation.FuncAnimation(fig, update_plot,
    #                                                frames=frame_amount, interval=2,
    #                                                repeat=True, blit=True)
    # plt.show()

    # strategy_name = f"test_martingale_noSafetyNet_noLiftSL_3Years_tp_{ctx.pct_tp*100}"
    strategy_name = f"pure_grid_{SYMBOL}"
    print(strategy_name)



    stats_df, winrate, apr,prom,total_trade,max_absolute_drawdown,RoMAD,sharpe_ratio,pct_positive_months,pct_positive_week,offset_total = get_stats(trade_df=trade_df, summary_df=summary_df,trades_bt=trades_bt)
    time_index = pd.to_datetime(summary_df["time"])
    fig, ax1 = plt.subplots()
    ax1.plot(time_index, summary_df["port_value"], label="port value", color="green")
    # ax1.plot(time_index, summary_df["benchmark"], label="benchmark",color="black",linestyle='--')
    # ax1.legend(loc="lower right")
    # ax1.legend(loc="lower right")
    # ax2 = ax1.twinx()
    # ax2.plot(time_index, summary_df["last_price"], label="price")
    # ax1.legend(loc="upper right")
    # ax2.legend(loc="lower right")
    plt.title(strategy_name)
    plt.show()
    # pprint(trades_bt)
    writer = pd.ExcelWriter(f"{strategy_name}.xlsx", engine="xlsxwriter")
    stats_df.to_excel(writer, sheet_name="statistics_data")
    trade_df.to_excel(writer, sheet_name="trade")
    summary_df.to_excel(writer, sheet_name="summary")
    positions_df.to_excel(writer, sheet_name="position")
    writer.close()