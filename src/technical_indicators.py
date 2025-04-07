"""
This module provides functions to calculate various technical indicators.
Indicators include ATR, RSI, MACD, SMA, VWAP, ADX, CCI, Stochastic Oscillator, and more.
The functions are designed to operate on historical price data (high, low, close, volume) using pandas Series ().

Additionally, this module contains a class `TechnicalIndicators` that encapsulates the calculation of these indicators 
and allows for streamlined processing of financial data.
"""

import numpy as np
import pandas as pd

class TechnicalIndicators:
    """
    A class to calculate various technical indicators as input to our LSTM.
    
    This class provides methods to compute:
    - Average True Range (ATR)
    - Relative Strength Index (RSI)
    - Stochastic Oscillator (%K)
    - Simple Moving Average (SMA)
    - Volume Weighted Average Price (VWAP)
    - Moving Average Convergence Divergence (MACD)
    - Average Directional Index (ADX)
    - Commodity Channel Index (CCI)
    - Williams %R
    - Parabolic SAR
    - On-Balance Volume (OBV)
    - Chaikin Money Flow (CMF)

    Attributes:
        high (pd.Series): A pandas Series containing the high prices for each period.
        low (pd.Series): A pandas Series containing the low prices for each period.
        close (pd.Series): A pandas Series containing the closing prices for each period.
        volume (pd.Series): A pandas Series containing the trading volumes for each period.
    
    Methods:
        calculate_atr(period=14): Calculates the Average True Range (ATR).
        calculate_rsi(period=14): Calculates the Relative Strength Index (RSI).
        calculate_stochastic_oscillator(period=14): Calculates the Stochastic Oscillator (%K).
        calculate_sma(period): Calculates the Simple Moving Average (SMA).
        calculate_vwap(): Calculates the Volume Weighted Average Price (VWAP).
        calculate_macd(fast_period=12, slow_period=26, signal_period=9): Calculates the Moving Average Convergence Divergence (MACD).
        calculate_adx(period=14): Calculates the Average Directional Index (ADX).
        calculate_cci(period=14): Calculates the Commodity Channel Index (CCI).
        calculate_williams_r(period=14): Calculates the Williams %R.
        calculate_parabolic_sar(initial_af=0.02, max_af=0.2, period=14): Calculates the Parabolic SAR (Stop and Reverse).
        calculate_obv(): Calculates the On-Balance Volume (OBV).
        calculate_cmf(period=20): Calculates the Chaikin Money Flow (CMF).
        process_all(): Calculates all technical indicators and returns a DataFrame containing them.
    """

    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series):
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def calculate_atr(self, period: int = 14) -> pd.Series:
        """
        Calculates the Average True Range (ATR).
        """
        # Reverse the order so the oldest date comes first
        high_series = self.high.iloc[::-1]
        low_series = self.low.iloc[::-1]
        close_series = self.close.iloc[::-1]

        # Compute True Range
        high_low = high_series - low_series
        high_close = (high_series - close_series.shift()).abs()
        low_close = (low_series - close_series.shift()).abs()

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))

        # Compute ATR on reversed data
        atr = true_range.rolling(window=period, min_periods=1).mean()

        # Reverse the result back to match the original order
        return atr.iloc[::-1]


    def calculate_rsi(self, period: int = 14) -> pd.Series:
        # Reverse the series so that the oldest date comes first
        close_series = self.close.iloc[::-1]
        
        # Calculate the change in closing price
        delta = close_series.diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate rolling averages (you might want min_periods=period 
        # if you require a full window before computing an RSI)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        # Calculate the relative strength and then the RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Set the first period-1 values to NaN because there's not enough data,
        # which now will be at the bottom of the reversed series.
        rsi.iloc[:period - 1] = np.nan
        
        # Reverse the result back to the original order (newest first)
        return rsi.iloc[::-1]


    def calculate_stochastic_oscillator(self, period: int = 14) -> pd.Series:
        """
        Calculates the Stochastic Oscillator (%K).
        """
        # Reverse the series so that the oldest date comes first
        low_series = self.low.iloc[::-1]
        high_series = self.high.iloc[::-1]
        close_series = self.close.iloc[::-1]
        
        # Calculate the rolling minimum and maximum on reversed data
        lowest_low = low_series.rolling(window=period, min_periods=1).min()
        highest_high = high_series.rolling(window=period, min_periods=1).max()
        
        # Calculate the Stochastic Oscillator on reversed data
        stochastic_k = 100 * (close_series - lowest_low) / (highest_high - lowest_low)
        
        # Reverse the result back to the original order
        return stochastic_k.iloc[::-1]


    def calculate_sma(self, period: int) -> pd.Series:
        """
        Calculates the Simple Moving Average (SMA).
        """
        # Reverse the series so that the oldest date comes first
        close_series = self.close.iloc[::-1]

        # Compute the moving average on reversed data
        sma = close_series.rolling(window=period, min_periods=1).mean()

        # Reverse the result back to match the original order
        return sma.iloc[::-1]


    def calculate_vwap(self) -> pd.Series:
        """
        Calculates the Volume Weighted Average Price (VWAP).
        """
        typical_price = (self.high + self.low + self.close) / 3
        vwap = (typical_price * self.volume).cumsum() / self.volume.cumsum()

        return vwap

    def calculate_macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Calculates the Moving Average Convergence Divergence (MACD).
        """
        # Reverse the closing prices to be in chronological order
        close_series = self.close.iloc[::-1]

        # Compute EMAs on the correctly ordered data
        fast_ema = close_series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = close_series.ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD on reversed data
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()

        # Reverse the results back to the original order (most recent first)
        return macd.iloc[::-1], signal_line.iloc[::-1]


    def calculate_adx(self, period: int = 14) -> pd.Series:
        """
        Calculates the Average Directional Index (ADX).
        """
        # Reverse the data so the oldest date comes first
        high = self.high.iloc[::-1]
        low = self.low.iloc[::-1]
        close = self.close.iloc[::-1]
        
        # Calculate True Range (TR)
        tr = pd.concat([
            high - low, 
            (high - close.shift()).abs(), 
            (low - close.shift()).abs()
        ], axis=1)
        true_range = tr.max(axis=1)

        # Calculate directional movements
        plus_dm = high.diff()
        minus_dm = low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        # Smooth the values using rolling sums
        smoothed_plus_dm = plus_dm.rolling(window=period, min_periods=1).sum()
        smoothed_minus_dm = minus_dm.rolling(window=period, min_periods=1).sum()
        smoothed_tr = true_range.rolling(window=period, min_periods=1).sum()

        # Calculate the directional indicators (DI)
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)

        # Calculate ADX as the smoothed average of the absolute difference between DI's
        adx = 100 * ((plus_di - minus_di).abs().rolling(window=period, min_periods=1).mean())

        # Reverse the result back to the original order (newest first)
        return adx.iloc[::-1]


    def calculate_cci(self, period: int = 14) -> pd.Series:
        """
        Calculates the Commodity Channel Index (CCI).
        """
        # Reverse the series so the oldest date comes first
        high_series = self.high.iloc[::-1]
        low_series = self.low.iloc[::-1]
        close_series = self.close.iloc[::-1]

        # Compute the typical price on reversed data
        typical_price = (high_series + low_series + close_series) / 3

        # Compute moving average and mean deviation on reversed data
        moving_average = typical_price.rolling(window=period, min_periods=1).mean()
        mean_deviation = typical_price.subtract(moving_average).abs().rolling(window=period, min_periods=1).mean()

        # Compute CCI on reversed data
        cci = (typical_price - moving_average) / (0.015 * mean_deviation)

        # Reverse the result back to the original order
        return cci.iloc[::-1]


    def calculate_williams_r(self, period: int = 14) -> pd.Series:
        """
        Calculates the Williams %R.
        """
        # Reverse the series so the oldest date comes first
        high_series = self.high.iloc[::-1]
        low_series = self.low.iloc[::-1]
        close_series = self.close.iloc[::-1]

        # Compute rolling highest high and lowest low on reversed data
        highest_high = high_series.rolling(window=period, min_periods=1).max()
        lowest_low = low_series.rolling(window=period, min_periods=1).min()

        # Compute Williams %R on reversed data
        williams_r = (highest_high - close_series) / (highest_high - lowest_low) * -100

        # Reverse the result back to the original order
        return williams_r.iloc[::-1]


    def calculate_parabolic_sar(self, initial_af: float = 0.02, max_af: float = 0.2, period: int = 14) -> pd.Series:
        """
        Calculates the Parabolic SAR (Stop and Reverse) indicator.
        """
        sar = pd.Series(index=self.high.index)
        af = initial_af
        trend = 1
        ep = self.high[0]
        sar[0] = self.low[0]

        for i in range(1, len(self.high)):
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

            if trend == 1:
                sar[i] = min(sar[i], self.low[i - 1], self.low[i])
            else:
                sar[i] = max(sar[i], self.high[i - 1], self.high[i])

            if trend == 1 and self.high[i] < sar[i]:
                trend = -1
                ep = self.low[i]
                af = initial_af
                sar[i] = ep
            elif trend == -1 and self.low[i] > sar[i]:
                trend = 1
                ep = self.high[i]
                af = initial_af
                sar[i] = ep

            if trend == 1:
                ep = max(ep, self.high[i])
                af = min(af + initial_af, max_af)
            else:
                ep = min(ep, self.low[i])
                af = min(af + initial_af, max_af)

        return sar

    def calculate_obv(self) -> pd.Series:
        """
        Calculates the On-Balance Volume (OBV).
        """
        # Reverse the data so the oldest date comes first
        close_series = self.close.iloc[::-1]
        volume_series = self.volume.iloc[::-1]

        # Initialize OBV
        obv = pd.Series(index=close_series.index, dtype=float)
        obv.iloc[0] = 0  # Start at 0

        # Compute OBV using the correct order
        for i in range(1, len(close_series)):
            if close_series.iloc[i] > close_series.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume_series.iloc[i]
            elif close_series.iloc[i] < close_series.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume_series.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        # Reverse the result back to the original order
        return obv.iloc[::-1]


    def calculate_cmf(self, period: int = 20) -> pd.Series:
        """
        Calculates the Chaikin Money Flow (CMF).
        """
        # Reverse the series so the oldest date comes first
        high_series = self.high.iloc[::-1]
        low_series = self.low.iloc[::-1]
        close_series = self.close.iloc[::-1]
        volume_series = self.volume.iloc[::-1]

        # Compute money flow multiplier on reversed data
        money_flow_multiplier = ((close_series - low_series) - (high_series - close_series)) / (high_series - low_series)

        # Compute money flow volume on reversed data
        money_flow_volume = money_flow_multiplier * volume_series

        # Compute rolling sums on reversed data
        cmf = money_flow_volume.rolling(window=period, min_periods=1).sum() / volume_series.rolling(window=period, min_periods=1).sum()

        # Reverse the result back to the original order
        return cmf.iloc[::-1]


    def process_all(self) -> pd.DataFrame:
        """
        Calculates all technical indicators and returns them in a DataFrame.
        """
        indicators = {
            'ATR': self.calculate_atr(),
            'RSI': self.calculate_rsi(),
            'Stochastic Oscillator': self.calculate_stochastic_oscillator(),
            'SMA': self.calculate_sma(14),
            'VWAP': self.calculate_vwap(),
            'MACD': self.calculate_macd()[0],  # Returning MACD line
            'ADX': self.calculate_adx(),
            'CCI': self.calculate_cci(),
            'Williams %R': self.calculate_williams_r(),
            'Parabolic SAR': self.calculate_parabolic_sar(),
            'OBV': self.calculate_obv(),
            'CMF': self.calculate_cmf()
        }

        # Combine all indicators into a single DataFrame
        df = pd.concat(indicators, axis=1)
        return df


# Main block to write technical indicators to TQQQ_data.csv for training
if __name__ == "__main__":
    
    df = pd.read_csv("data/TQQQ_data.csv")
    
    TQQQ_close = df["TQQQ Close"][0:1240]
    TQQQ_high = df["TQQQ High"][0:1240]
    TQQQ_low = df["TQQQ Low"][0:1240]
    TQQQ_volume = df["TQQQ Volume (M)"][0:1240]

    tec_i = TechnicalIndicators(TQQQ_high, TQQQ_low, TQQQ_close, TQQQ_volume)

    tec_i.process_all().to_csv("data/TQQQ_data_aug.csv")
    print("Processed all indicators and saved to data/TQQQ_data_aug.csv.")