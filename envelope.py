import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def gauss(x, h):
    return np.exp(-(x ** 2) / (h * h * 2))

ticker = 'BTC-USD'
data = yf.download(ticker, start='2024-01-01', end='2024-08-17')
close = data['Close']

h = 7.0  # Bandwidth
mult = 3.0  # Multiplier
repaint = True  # Repainting mode

n = len(close)
mae_values = []
upper_band = []
lower_band = []

if not repaint:
    coefs = [gauss(i, h) for i in range(500)]
    den = sum(coefs)
    out = sum(close[i] * coefs[i] for i in range(500)) / den
    mae = close.rolling(window=499).apply(lambda x: np.abs(x - out).mean(), raw=True) * mult
    upper_band = out + mae
    lower_band = out - mae
else:
    nwe = []
    sae = 0.0
    for i in range(min(499, n - 1)):
        sum_ = 0.0
        sumw = 0.0
        for j in range(min(499, n - 1)):
            w = gauss(i - j, h)
            sum_ += close[j] * w
            sumw += w
        y2 = sum_ / sumw
        sae += np.abs(close[i] - y2)
        nwe.append(y2)

    sae /= min(499, n - 1)
    sae *= mult

    for y2 in nwe:
        upper_band.append(y2 + sae)
        lower_band.append(y2 - sae)

plt.figure(figsize=(14, 7))
plt.plot(close.index, close, label='Close', color='black')
if repaint:
    plt.plot(close.index[-len(upper_band):], upper_band, label='Upper', color='teal')
    plt.plot(close.index[-len(lower_band):], lower_band, label='Lower', color='red')
else:
    plt.plot(close.index, upper_band, label='Upper', color='teal')
    plt.plot(close.index, lower_band, label='Lower', color='red')
plt.legend()
plt.title(f'{ticker} - Envelope')
plt.xlabel('Date')
plt.ylabel('Price')
plt.yscale('log')
plt.grid(True)
plt.show()
