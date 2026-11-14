# Runs the book's __main__ block from mac.py unchanged; only csv_dir (data source) is supplied.
import sys, datetime
sys.path.insert(0, "book_code")
from backtest import Backtest
from data import HistoricCSVDataHandler
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from mac import MovingAverageCrossStrategy
csv_dir = sys.argv[1]; symbol_list = sys.argv[2].split(",")
initial_capital = 100000.0
start_date = datetime.datetime(1990,1,1,0,0,0)
heartbeat = 0.0
backtest = Backtest(csv_dir, symbol_list, initial_capital, heartbeat, start_date,
                    HistoricCSVDataHandler, SimulatedExecutionHandler, Portfolio, MovingAverageCrossStrategy)
backtest.simulate_trading()
