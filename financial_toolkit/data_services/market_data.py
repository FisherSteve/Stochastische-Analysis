import yfinance as yf
import pandas as pd
import numpy as np
try:
    from arch import arch_model # For GARCH
except ImportError:
    print("Warning: 'arch' library not found. GARCH functionality will not be available. "
          "Please install it by running: pip install arch")
    arch_model = None # Placeholder if not installed

def get_historical_stock_data(ticker_symbols, start_date, end_date, interval="1d"):
    """
    Fetches historical stock data from Yahoo Finance using yf.download.
    Can fetch for single or multiple tickers.
    With auto_adjust=True, the 'Close' column is typically the dividend and split adjusted closing price.

    Args:
        ticker_symbols (str or list): The stock ticker symbol(s) (e.g., "AAPL", ["MSFT", "GOOGL"]).
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        interval (str): Data interval. Default is "1d".

    Returns:
        pandas.DataFrame: DataFrame containing historical data.
                          If single ticker, columns are Open, High, Low, Close, Volume.
                          If multiple tickers, columns are a MultiIndex: (Metric, TickerSymbol).
                          Returns None if data fetching fails or no data is found.
    """
    try:
        hist_data = yf.download(
            tickers=ticker_symbols, 
            start=start_date, 
            end=end_date, 
            interval=interval, 
            auto_adjust=True, 
            progress=False,
            group_by='ticker' if isinstance(ticker_symbols, list) and len(ticker_symbols) > 1 else 'column'
        )
        
        if hist_data.empty:
            print(f"No data found for {ticker_symbols} from {start_date} to {end_date} with interval {interval}.")
            return None
        
        if isinstance(ticker_symbols, list) and len(ticker_symbols) > 1:
            # yf.download with list of tickers and group_by='ticker' will give MultiIndex like ('AAPL', 'Close')
            # To change to ('Close', 'AAPL') for easier selection data['Close']['AAPL']:
            hist_data = hist_data.stack(level=0).unstack(level=1) # Swaps levels
            # Or, if the default for multiple tickers is already ('Metric', 'Ticker'), then no swap needed.
            # The standard return for yf.download(tickers=list_of_tickers) is ('Adj Close', 'AAPL'), etc.
            # Let's assume yf.download returns ('Metric', 'Ticker') when group_by is not specified or is 'column' by default with list
            # Re-fetch with default grouping for multiple tickers to get ('Close', 'AAPL') etc.
            hist_data_std_multi = yf.download(
                tickers=ticker_symbols, start=start_date, end=end_date, interval=interval,
                auto_adjust=True, progress=False
            )
            if hist_data_std_multi.empty: return None
            return hist_data_std_multi
        elif isinstance(ticker_symbols, str): # Single ticker already flat
             return hist_data

        return hist_data
    
    except Exception as e:
        print(f"An error occurred while fetching data for {ticker_symbols}: {e}")
        return None

def calculate_historical_volatility(price_series, window=252, clean_data=True):
    if not isinstance(price_series, pd.Series):
        if isinstance(price_series, pd.DataFrame) and price_series.shape[1] == 1:
            price_series = price_series.squeeze()
        else:
            raise TypeError(f"price_series must be a pandas Series. Got {type(price_series)}.")
            
    if price_series.empty or len(price_series) < 2: 
        return np.nan

    log_returns = np.log(price_series / price_series.shift(1))
    
    if clean_data:
        log_returns = log_returns.dropna()

    if len(log_returns) < 2 : 
        return np.nan
        
    daily_volatility = log_returns.std() 
    annualized_volatility = daily_volatility * np.sqrt(window)
    
    return annualized_volatility

def calculate_historical_correlation_matrix(data_df_multi_ticker, price_metric='Close'):
    if not isinstance(data_df_multi_ticker, pd.DataFrame) or data_df_multi_ticker.empty:
        print("Input data_df_multi_ticker must be a non-empty pandas DataFrame.")
        return None

    close_prices_df = None
    if isinstance(data_df_multi_ticker.columns, pd.MultiIndex):
        try:
            # Assumes MultiIndex is ('Metric', 'Ticker')
            if price_metric in data_df_multi_ticker.columns.levels[0]:
                 close_prices_df = data_df_multi_ticker[price_metric]
            else: # Try swapping levels if ('Ticker', 'Metric')
                if price_metric in data_df_multi_ticker.columns.levels[1]:
                    temp_df = data_df_multi_ticker.copy()
                    temp_df.columns = temp_df.columns.swaplevel(0,1)
                    close_prices_df = temp_df[price_metric]
                else:
                    print(f"Price metric '{price_metric}' not found in MultiIndex columns. Levels: {data_df_multi_ticker.columns.levels}")
                    return None
        except KeyError:
            print(f"Price metric '{price_metric}' not found. Available metrics: {data_df_multi_ticker.columns.levels[0]}")
            return None
    else: # Flat DataFrame, assumed to be single ticker, not suitable for correlation matrix directly
        print("Data for correlation matrix expected MultiIndex columns ('Metric', 'Ticker').")
        return None

    if close_prices_df is None or close_prices_df.empty or close_prices_df.shape[1] < 2:
        print("Not enough ticker data in selected price metric to calculate correlation matrix.")
        return None

    log_returns_df = np.log(close_prices_df / close_prices_df.shift(1))
    log_returns_df = log_returns_df.dropna() 

    if len(log_returns_df) < 2 or log_returns_df.shape[1] < 2:
        print("Not enough overlapping log return data for multiple tickers to calculate correlation.")
        return None
        
    correlation_matrix = log_returns_df.corr()
    return correlation_matrix

def fit_garch_and_forecast_volatility(price_series, forecast_horizon_days=21, p=1, q=1, trading_days_per_year=252):
    """
    Fits a GARCH(p,q) model to log returns and forecasts average annualized volatility.

    Args:
        price_series (pd.Series): Series of historical prices (e.g., 'Adj Close' or 'Close').
        forecast_horizon_days (int): Number of days ahead to forecast volatility.
        p (int): Order of GARCH term.
        q (int): Order of ARCH term.
        trading_days_per_year (int): For annualization.

    Returns:
        float: Average annualized forecasted volatility over the horizon.
               Returns np.nan if GARCH fitting or forecasting fails.
    """
    if arch_model is None:
        print("GARCH modeling requires the 'arch' library. It's not installed.")
        return np.nan
    if not isinstance(price_series, pd.Series):
        raise TypeError("price_series must be a pandas Series.")
    if len(price_series) < max(p,q) + 5: # Need some data
        print("Price series too short for GARCH fitting.")
        return np.nan

    # Calculate percentage log returns and scale (GARCH often works better with % returns)
    log_returns = 100 * np.log(price_series / price_series.shift(1)).dropna()
    
    if log_returns.empty:
        print("No log returns available for GARCH fitting.")
        return np.nan

    try:
        # Fit GARCH(p,q) model - common choice is GARCH(1,1)
        # Use 'ConstantMean' as returns often have a small, non-zero mean.
        # Volatility model is 'GARCH'. Distribution can be 'Normal' or 't' for fatter tails.
        model = arch_model(log_returns, vol='Garch', p=p, q=q, mean='Constant', dist='Normal')
        # disp='off' suppresses convergence output
        results = model.fit(disp='off', show_warning=False) 
        
        # Forecast conditional variance for the horizon
        # forecast() returns an ARCHModelForecast object
        forecast = results.forecast(horizon=forecast_horizon_days, reindex=False) # reindex=False for simpler output
        
        # Get the h.1, h.2, ... h.forecast_horizon_days variance forecasts
        # The forecast object structure can be a bit nested.
        # For single series, it's usually forecast.variance.iloc[-1].values
        future_variance_forecasts = forecast.variance.iloc[-1].values # Daily percentage variances
        
        if len(future_variance_forecasts) == 0:
            print("GARCH forecast produced no variance values.")
            return np.nan

        # Average of the forecasted daily percentage variances
        avg_future_daily_perc_variance = np.mean(future_variance_forecasts)
        
        # Convert back to daily volatility (sqrt) and de-scale from percentage
        avg_future_daily_volatility = np.sqrt(avg_future_daily_perc_variance) / 100.0
        
        # Annualize
        annualized_forecasted_volatility = avg_future_daily_volatility * np.sqrt(trading_days_per_year)
        
        return annualized_forecasted_volatility
        
    except Exception as e:
        print(f"Error during GARCH fitting or forecasting: {e}")
        return np.nan


if __name__ == '__main__':
    print("--- Example: Fetching Multi-Ticker data, Volatility, Correlation, GARCH ---")
    tickers_main = ["AAPL", "MSFT"]
    start_main = "2022-01-01" 
    end_main = pd.Timestamp.today().strftime("%Y-%m-%d") 
    
    multi_data_main = get_historical_stock_data(tickers_main, start_main, end_main)
    
    if multi_data_main is not None and not multi_data_main.empty:
        price_metric_main = 'Close' # Assuming 'Close' is adjusted by yfinance auto_adjust
        
        # Check if expected column structure exists
        valid_data_structure = False
        if isinstance(multi_data_main.columns, pd.MultiIndex) and \
           price_metric_main in multi_data_main.columns.levels[0]:
            valid_data_structure = True
        elif not isinstance(multi_data_main.columns, pd.MultiIndex) and \
             price_metric_main in multi_data_main.columns and len(tickers_main)==1: # Single ticker case
            # If single ticker, wrap it for consistency in loop
            multi_data_main.columns = pd.MultiIndex.from_product([[price_metric_main, 'Open', 'High', 'Low', 'Volume'], [tickers_main[0]]])
            multi_data_main = multi_data_main.stack().unstack(0) # Pivot to match ('Metric', 'Ticker')
            valid_data_structure = True


        if valid_data_structure and price_metric_main in multi_data_main.columns.levels[0]:
            print(f"\nFetched data (showing head of '{price_metric_main}' prices):")
            print(multi_data_main[price_metric_main].head())

            for ticker_sym_main in tickers_main:
                if ticker_sym_main in multi_data_main[price_metric_main].columns:
                    prices = multi_data_main[price_metric_main][ticker_sym_main].dropna()
                    if not prices.empty:
                        vol_hist = calculate_historical_volatility(prices)
                        vol_garch = fit_garch_and_forecast_volatility(prices, forecast_horizon_days=63) # Forecast 3 months
                        
                        if not np.isnan(vol_hist): print(f"  Hist. Vol for {ticker_sym_main}: {vol_hist:.4f}")
                        else: print(f"  Hist. Vol for {ticker_sym_main}: Not calculable")
                        if not np.isnan(vol_garch): print(f"  GARCH Forecasted Vol for {ticker_sym_main}: {vol_garch:.4f}")
                        else: print(f"  GARCH Vol for {ticker_sym_main}: Not calculable")
                else:
                    print(f"'{price_metric_main}' data for {ticker_sym_main} not found.")
            
            if len(tickers_main) > 1:
                 print("\nCalculating correlation matrix...")
                 corr_matrix_main = calculate_historical_correlation_matrix(multi_data_main, price_metric=price_metric_main)
                 if corr_matrix_main is not None:
                     print("Correlation Matrix:")
                     print(corr_matrix_main)
        else:
            print(f"'{price_metric_main}' data not structured as expected. Columns: {multi_data_main.columns}")
    else:
        print(f"Could not fetch multi-ticker data for {tickers_main}.")
