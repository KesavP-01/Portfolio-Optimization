import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt 
import matplotlib.pyplot as plt



tickers = ['UNH', 'MSFT', 'GS', 'HD', 'CAT', 'MCD', 'CRM', 'AMGN', 'TRV']

data = {}
start = dt.datetime(2023,2,21)
end = dt.datetime(2024,2,21)
close = []

for ticker in tickers:
    df = yf.download(ticker,start, end, period = '1d' )
    data[ticker] = df['Adj Close']
    close = pd.DataFrame.from_dict(data)
    
returns = close.pct_change()

var_matrix = returns.cov()


port_weights_list = []
port_variance_list = []
port_voltality_list = []
port_returns_list = []

num_assets = len(returns.columns)
num_port = 10000

ind_returns = close.resample('Y').last().pct_change().mean()

for port in range(num_port):
    port_weight = np.random.random(num_assets)
    port_weights = port_weight/np.sum(port_weight)
    portfolio_returns = ind_returns.dot(port_weights)
    port_variance = np.transpose(port_weights)@var_matrix@port_weights
    port_voltatlity = np.sqrt(port_variance)
    ann_volatility = port_voltatlity*(np.sqrt(250))
    port_weights_list.append(port_weights)
    port_variance_list.append(port_variance)
    port_voltality_list.append(ann_volatility)
    port_returns_list.append(portfolio_returns)
    
final = {'Volatility' : port_voltality_list, 'Returns' : port_returns_list}

for counter, symbol in enumerate(close.columns.tolist()):
    final[symbol + 'weights'] = [w[counter] for w in port_weights_list]

final_data = pd.DataFrame.from_dict(final)


min_vol = final_data.iloc[final_data['Volatility'].idxmin()]

rf = 0.01
optimal_risk = final_data.iloc[((final_data['Returns']- rf)/final_data['Volatility']).idxmax()]   

final_data.plot.scatter(x = 'Volatility', y = 'Returns', marker = 'o', color = 'g', s=10, alpha = 0.5, grid = True, figsize = (8,8))
plt.scatter(min_vol[0], min_vol[1], color = 'b', marker = '*', s = 500)
plt.scatter(optimal_risk[0], optimal_risk[1], color = 'y', marker = '*', s = 500)
plt.xlabel('Risk')
plt.ylabel('Returns')



    
