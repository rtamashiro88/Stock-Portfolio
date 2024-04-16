'''
Module: Data API Functions
Created On: 2019-07-05              Last Modified: 2019-08-07
**************************************************************************
    **READ ME**

    [Description]
    Module Contains Custom Functions For The Below List of Registerd APIs and
    Returning Specific Financial Data For The Specified Stock Symbol/Company

    [Finance APIs]
    SEC Edgar   Security Exchange Commission - Company Filings & Financial Disclosures
                    -   Consolidated Financial Statements (JSON)
                    -   Performance Ratios
                    -   Insider Trades

    Quandl      Stock Performance Data - Time Series Data
                    -   Open, Low, High, Close, Volume, Ex-Dividend, Splits


'''
import datetime as dt
import pandas as pd
import numpy as np
import quandl as ql
from config import quandl_api_key


'''
    Quandl API - Data Collection Functions

'''


def quandl_stock_data(symbol, verbose=False):
    # <Define> DataFrame Column Headers
    headers = [
        'Open',
        'High',
        'Low',
        'Close',
        'Volume',
    ]

    # <Set> API Query Parameters
    query_params = {
        'symbol': symbol.upper(),
        'start_date': "2014-01-01",
        "end_date": "2019-01-01",
        "collapse": "monthly",
        "data_type": "pandas",    # [numpy | pandas ] Array vs DataFrame
    }

    try:
        stock_returns = ql.get(
            f"WIKI/{query_params['symbol']}",
            start_date=query_params['start_date'],
            end_date=query_params['end_date'],
            colapse=query_params['collapse'],
            returns=query_params['data_type'],
            authtoken=quandl_api_key
        )[headers]

        if verbose:
            print(f"\n[Quandl] Query API Summary:\n")
            print("-" * 75, "\n")
            for param, val in query_params.items():
                print(f"- {param}:", val)

            print("\n", ("-" * 75), "\n")
            print("\n[Preview] Response DataFrame\n")
            print("\n", stock_returns.head(10), "\n")
            print("-" * 75, "\n")
            print("\n[View] DataFrame Columns -- Data Uniformity\n")
            print(stock_returns.count(), "\n")
            print("-" * 75, "\n")
            print("\n[View] DataFrame Columns -- Data Types\n")
            print(stock_returns.dtypes, "\n")

        return stock_returns

    except ql.NotFoundError:
        print(f"\n[Error | API Query] Invalid Company Symbol: {query_params['symbol']}")
        return None

# Portfolio Optimization Function


def optimize_portfolio(assets, simulations=5000):
    num_assets = len(assets)
    portfolio = closing_prices(assets[0])
    print(f'[{0}] Retrieving Stock Data: {assets[0].upper()}')

    for i, asset in enumerate(assets[1:]):
        print(f'[{i + 1}] Retrieving Stock Data: {asset}')
        add_stock = closing_prices(asset)
        portfolio = pd.merge(portfolio, add_stock, on="Date", how="inner")
        del add_stock

    portfolio.set_index("Date", inplace=True)

    print(f'\nOptimizing Portfolio Weights >> Simulations: x {simulations}')

    # Monte Carlo Simulation
    portfolio_log = []
    portfolio_sim = {}
    for i in range(simulations):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        WTSp = zip(assets, weights)
        RTNp = exp_portfolio_return(portfolio, weights)
        VARp = exp_portfolio_variance(portfolio, weights)

        portfolio_sim = {a: round(wt, 4) for a, wt in WTSp}
        portfolio_sim["Return"] = RTNp
        portfolio_sim["Variance"] = VARp
        portfolio_sim["Sharpe"] = mod_sharpe_ratio(RTNp, VARp)
        portfolio_log.append(portfolio_sim)

    log_df = pd.DataFrame(portfolio_log)
    ranked_df = log_df.sort_values("Sharpe", ascending=False)

    print(f'\nOptimized Portfolio Weights:')
    print(ranked_df.iloc[0])
    return dict(ranked_df.iloc[0])


# Portfolio Performance Back-Testing Function


def backtest_portfolio(pfolio):
    exclude = ["Return", "Sharpe", "Variance"]
    assets = [(a, wt) for a, wt in pfolio.items() if a not in exclude]

    # Initialize Portfolio Back-Test Performance DataFrame
    back_test = closing_prices(assets[0][0]).set_index("Date")
    back_test = np.log(back_test / back_test.shift(1)).iloc[1:]
    back_test = back_test.apply(lambda x: x * assets[0][1])
    print(f'\nTicker: {assets[0][0]} \tPortfolio Weight: {assets[0][1]}')
    print(back_test.head())

    for allocation in assets[1:]:
        stock = allocation[0]
        weight = allocation[1]
        print(f'\nTicker: {stock} \tPortfolio Weight: {weight}')

        closing_data = closing_prices(stock).set_index("Date")
        pct_return = np.log(closing_data / closing_data.shift(1)).iloc[1:]
        pct_return = pct_return.apply(lambda x: x * weight)
        back_test = pd.merge(back_test, pct_return, on="Date", how="inner")
        print(pct_return.head())

    back_test["RTNp"] = back_test.sum(axis=1)
    print("\n[Historic] Portfolio Performance:\n", back_test.head())

    return back_test


# Portfolio Performance Evaluation Function


def evaluate_portfolio(rtns):
    RTNm = pd.read_csv("S&P500.csv")[["Date", "Close"]]
    RTNm["Date"] = pd.to_datetime(RTNm["Date"])
    RTNm = RTNm.rename(columns={"Close": "RTNm"}).set_index("Date")
    RTNm = np.log(RTNm / RTNm.shift(1)).iloc[1:]

    rtns = pd.merge(rtns, RTNm, on="Date", how="inner")
    rtns["Excess"] = rtns["RTNp"] - rtns["RTNm"]
    rtns["Compare"] = rtns["Excess"] > 0
    rtns["Compare"] = rtns["Compare"].apply(lambda x: "Outperform" if x else "Underperform")
    print(rtns.head())

    return rtns


'''
Category: Machine Learning Model Training

Functions:
    0.  setup_ml_training_data_db
    1.  generate_random_portfolio
    2.  compile_random_portfolio
    2.  generate_ml_training_data
    3.  update_ml_training_database

Description:
    General Utility Functions to Generate/Simulate Random Portflios Which Are Later
    Used to Train our Neural Network Machine Learning Training Model. The ML Data
    Generation Process Occurs Over Four Steps

        1.  Initialize & Configure DB:
                Using Python A MySQL/SQLite Database is Initialized and Configured
                in Order to Store the Simulated Portfolio Data That is Generated
                Which We Later Use to Train Our Machine Learning Model

        2.  Generate Random Test Portfolios:
                [Input]  Params: List of Stock Ticker Symbols
                [Output] Return: DataFrame - 5 Year Historic Closing Prices

                Randomly Selects Between 5-10 Stocks in Which to Include in the
                Test Portfolio. Using API calls to Quandl (Database) it Pulls Historic
                Closing Price Data For the Past 5 Years and Aggregates it w/
                Associated Benchmark Performance into a DataFrame Which is Then
                Returned as the Functions Output.

        3.  Calculate Portfolio Statistics:
                [Input]  Params: DataFrame of Portflio Closing Prices
                [Output] Return: Dictionary Descriptive Portfolio Statistics

                Using the Returned Output From the Previous Function, Step 3
                Calculates the Following Portfolio Statistics:
                    a.  Counts Number of Stocks
                    b.  Portfolio Regresson Beta
                    c.  Portfolio Expected Return
                    d.  Portfolio Expected Variance
                    e.  Portfolio Sharpe Ratio

        4.  Update ML Training Database
                [Input]  Params: List of Dictiories - Portfolio Stats
                [Output] Return: None

'''

#   Random Test Portfolio Generation Function


def generate_random_portfolio():
    stocklist_df = pd.read_csv("StockTickers.csv")
    stocklist = list(dict(stocklist_df)["Tickers"])

    random_portfolio = []
    for i in range(random.randint(6, 10)):
        add_stock = random.choice(stocklist)
        if add_stock not in random_portfolio:
            random_portfolio.append(add_stock)
            del add_stock

    print("\nRandom Generated Portfolio", random_portfolio)
    return random_portfolio

#   Compile Random Portfolio Data -- Pull Historic Data & Aggregate to DataFrame


def compile_random_portfolio(p_stocks):
    print(f"\n<Quandl API> Stock Data: {p_stocks[0]}")
    sim_portfolio = closing_prices(p_stocks[0])
    for stock in p_stocks[1:]:
        print(f"<Quandl API> Stock Data: {stock}")
        add_stock = closing_prices(stock)
        sim_portfolio = pd.merge(sim_portfolio, add_stock, on="Date", how="inner")
        del add_stock

    benchmark = pd.read_csv("S&P500.csv")[["Date", "Close"]]
    benchmark["Date"] = pd.to_datetime(benchmark["Date"])
    benchmark = benchmark.rename(columns={"Close": "SP500"})

    output_portfolio = pd.merge(sim_portfolio, benchmark, on="Date", how="inner")

    print("\n[Output] Portfolio Closing Prices\n", output_portfolio.head())
    return output_portfolio.set_index("Date")

#   Portfolio Descriptive Statistics Calculation Function


def generate_ml_training_data(sim_portfolio):
    p_stocks = list(sim_portfolio.columns)
    p_stocks.remove("SP500")
    benchmark_portfolio = sim_portfolio["SP500"]
    stock_portfolio = sim_portfolio[p_stocks]

    print('\n[Training] Stock Portfolio\n', stock_portfolio.head())

    weights = np.random.random(len(p_stocks))
    weights /= np.sum(weights)

    p_allocation = [(pos[0], round(pos[1], 4)) for pos in zip(p_stocks, weights)]
    print("\n[Portfolio] Asset Allocation:\n", p_allocation)

    pct_returns = round(stock_portfolio.pct_change().iloc[1:], 4)
    pct_returns["RTNp"] = np.sum(pct_returns, axis=1)
    print("\n[Portfolio] Daily Returns:")
    print(pct_returns.head())

    # Calculate Portfolio Statistics (Return, Variance, Sharpe)
    p_rtn = exp_portfolio_return(stock_portfolio, weights)
    p_var = exp_portfolio_variance(stock_portfolio, weights)
    sp500_rtn = benchmark_portfolio.mean() * 250
    SP500_var = benchmark_portfolio.var() * 250

    mod_sharpe = mod_sharpe_ratio(p_rtn, p_var)

    unweighted_perform = round((stock_portfolio.iloc[-1] / stock_portfolio.iloc[0]) - 1, 4)
    print("\n[Portfolio] Unweighted Returns")
    print(unweighted_perform.head())

    weighted_perform = round(np.sum(unweighted_perform[p_stocks] * weights), 4)
    print(f"\n[Portfolio] Weighted Return: {weighted_perform}")

    benchmark_perform = round((benchmark_portfolio[-1] / benchmark_portfolio[0]) - 1, 4)

    # Portfolio Regression Beta Calculation
    p_beta = "N/A"
    print(f"\n[Benchmark | S&P500] Perfomance: {benchmark_perform}")

    # Need to Validate This Risk Adjusted Calculation
    pfolio_radj_perform = ((weighted_perform / p_var) > (benchmark_perform / sp500_var))
    pfolio_stats = {
        "CTp": len(p_stocks),
        "RTNp": round(p_rtn, 4),
        "VARp": round(p_var, 4),
        "SHRp": round(mod_sharpe, 4),
        "BETAp": p_beta,
        "SP500": round(benchmark_perform, 4),
        "PvSP": pfolio_radj_perform
    }

    return pfolio_stats

# Helper Functions - Optimize Portfolio, Backtest Portfolio Performance


def closing_prices(stock):
    price_data = quandl_stock_data(stock) \
        .rename(columns={"Close": stock.upper()})[stock.upper()] \
        .reset_index()
    return price_data


def exp_portfolio_return(portfolio, weights):
    log_returns = np.log(portfolio / portfolio.shift(1)).iloc[1:]
    return round(np.sum(weights * log_returns.mean()) * 250, 4)


def exp_portfolio_variance(portfolio, weights):
    log_returns = np.log(portfolio / portfolio.shift(1)).iloc[1:]
    return round(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 250, weights))), 4)


def mod_sharpe_ratio(ERp, EVARp):
    mkt_return = .098
    return round((ERp - mkt_return) / EVARp, 4)
