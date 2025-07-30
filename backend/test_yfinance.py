import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

# Set up session with retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Inject the session into yfinance
yf.shared._requests = session  # internal override

tickers = ["AAPL", "MSFT"]
print("Attempting to fetch 1-year historical data for:", ", ".join(tickers))

try:
    time.sleep(2)
    data = yf.download(
        tickers=tickers,
        period="1y",
        group_by='ticker',
        auto_adjust=False,
        progress=True,
        threads=False  # Avoid multi-threading if connection is unstable
    )

    for ticker in tickers:
        print(f"\nProcessing data for {ticker}...")
        try:
            ticker_data = data[ticker]
            if not ticker_data.empty:
                print(f"Successfully fetched data for {ticker}:")
                print(ticker_data.tail())
            else:
                print(f"No data found for {ticker}.")
        except KeyError:
            print(f"Data for {ticker} not found in the download response.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
