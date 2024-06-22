import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import seaborn as sns
import matplotlib.pyplot as plt

def download_data(ticker_list, start_date, end_date):
    data = {}
    start = start_date
    end = end_date
    for ticker in ticker_list:
        df = yf.download(ticker,start, end, period = '1d' )
        data[ticker] = df['Adj Close']
        main_data = pd.DataFrame.from_dict(data)
    return main_data

def optimization(data, no_of_portfolios):
    returns = data.pct_change()
    var_matrix = returns.cov()
    num_assets = len(data.columns)
    n = no_of_portfolios
    ind_returns = data.pct_change().mean() * 252
    
    port_weights_list = []
    port_variance_list = []
    port_voltality_list = []
    port_returns_list = []
    
    for port in range(n):
        port_weight = np.random.random(num_assets)
        port_weights = port_weight/np.sum(port_weight)
        portfolio_returns = ind_returns.dot(port_weights)
        port_variance = np.transpose(port_weights)@var_matrix@port_weights
        port_voltatlity = np.sqrt(port_variance)
        ann_volatility = port_voltatlity*(np.sqrt(252))
        port_weights_list.append(port_weights)
        port_variance_list.append(port_variance)
        port_voltality_list.append(ann_volatility)
        port_returns_list.append(portfolio_returns)
        
    final = {'Volatility' : port_voltality_list, 'Returns' : port_returns_list}

    for counter, symbol in enumerate(data.columns.tolist()):
        final[symbol + 'weights'] = [w[counter] for w in port_weights_list]

    optimized_data = pd.DataFrame.from_dict(final)
    return optimized_data

def volatility_optimization(data):
    num_assets = len(data.columns)
    bounds = Bounds(0, 1)
    constraint_matrix = np.ones((1, num_assets))
    linear_constraint = LinearConstraint(constraint_matrix, [1], [1])
    initial_weights = np.ones(num_assets) / num_assets
    ind_returns = data.pct_change().mean() * 252
    var_matrix = data.pct_change().cov()
    
    def port_vol(weights):
        return np.sqrt(np.dot(np.transpose(weights), np.dot(weights, var_matrix))*252)
    
    vol = minimize(port_vol, initial_weights, method= 'trust-constr', constraints= linear_constraint, bounds=bounds)
    min_wei = vol.x
    min_vol = port_vol(min_wei)
    port_returns = ind_returns.dot(min_wei)
    final_weight = np.transpose(min_wei)[np.newaxis]
    final_weights = pd.DataFrame(final_weight)
    final_weights.columns = stock_data.columns
    final_weights['Volatility'] = min_vol
    final_weights['Return'] = port_returns
    return final_weights

def Sharpe_ratio_optimization(data, risk_free_rate):
    num_assets = len(data.columns)
    bounds = Bounds(0, 1)
    rf = risk_free_rate
    constraint_matrix = np.ones((1, num_assets))
    linear_constraint = LinearConstraint(constraint_matrix, [1], [1])
    initial_weights = np.ones(num_assets) / num_assets
    ind_returns = data.pct_change().mean() * 252
    var_matrix = data.pct_change().cov()
    
    def port_vol(weights):
        return np.sqrt(np.dot(np.transpose(weights), np.dot(weights, var_matrix))*252)
    
    def inv_sharpe(weights):
        return  (np.sqrt(np.dot(np.transpose(weights), np.dot(weights, var_matrix))*252)) / (ind_returns.dot(weights) - rf) 
    
    opt = minimize(inv_sharpe, initial_weights, method='trust-constr', constraints= linear_constraint, bounds=bounds)
    min_wei = opt.x
    vol = port_vol(min_wei)
    returns = ind_returns.dot(min_wei)
    wei = np.transpose(min_wei)[np.newaxis]
    f_wei = pd.DataFrame(wei)
    f_wei.columns = data.columns
    f_wei['Volatility'] = vol
    f_wei['Return'] = returns
    return f_wei

    
    
tickers = ['RTO', 'NVO', 'KR', 'BURL', 'AGR', 'ARHS', 'ASX', 'AAPL', 'MSFT', 'META', 'DKNG', 'PLTR']
    
stock_data = download_data(tickers, '2023-03-01', '2024-03-01')
Optimized_portfolios = optimization(stock_data, 10000)
least_volatility = volatility_optimization(stock_data)
optimal_sharpe = Sharpe_ratio_optimization(stock_data, 0.01)


# Frontier Curve
plt.figure(figsize=(12,8), dpi=600)
sns.scatterplot(data=Optimized_portfolios, x= 'Volatility', y= 'Returns', hue= 'Returns', palette='viridis')
