import yfinance as yf

df = yf.download("AAPL", start="2023-01-01", end="2026-01-01", multi_level_index= False)
df.to_csv("data/aapl.csv")