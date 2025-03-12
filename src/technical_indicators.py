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
        high_low = self.high - self.low
        high_close = (self.high - self.close.shift()).abs()
        low_close = (self.low - self.close.shift()).abs()

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()

        return atr

    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """
        Calculates the Relative Strength Index (RSI).
        """
        delta = self.close.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_stochastic_oscillator(self, period: int = 14) -> pd.Series:
        """
        Calculates the Stochastic Oscillator (%K).
        """
        lowest_low = self.low.rolling(window=period).min()
        highest_high = self.high.rolling(window=period).max()

        stochastic_k = 100 * (self.close - lowest_low) / (highest_high - lowest_low)

        return stochastic_k

    def calculate_sma(self, period: int) -> pd.Series:
        """
        Calculates the Simple Moving Average (SMA).
        """
        return self.close.rolling(window=period).mean()

    def calculate_vwap(self) -> pd.Series:
        """
        Calculates the Volume Weighted Average Price (VWAP).
        """
        typical_price = (self.high + self.low + self.close) / 3
        vwap = (typical_price * self.volume).cumsum() / self.volume.cumsum()

        return vwap

    def calculate_macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.Series:
        """
        Calculates the Moving Average Convergence Divergence (MACD).
        """
        fast_ema = self.close.ewm(span=fast_period, adjust=False).mean()
        slow_ema = self.close.ewm(span=slow_period, adjust=False).mean()

        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()

        return macd, signal_line

    def calculate_adx(self, period: int = 14) -> pd.Series:
        """
        Calculates the Average Directional Index (ADX).
        """
        tr = pd.concat([self.high - self.low, 
                        (self.high - self.close.shift()).abs(), 
                        (self.low - self.close.shift()).abs()], axis=1)
        true_range = tr.max(axis=1)

        plus_dm = self.high.diff()
        minus_dm = self.low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        smoothed_plus_dm = plus_dm.rolling(window=period).sum()
        smoothed_minus_dm = minus_dm.rolling(window=period).sum()
        smoothed_tr = true_range.rolling(window=period).sum()

        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)

        adx = 100 * ((plus_di - minus_di).abs().rolling(window=period).mean())

        return adx

    def calculate_cci(self, period: int = 14) -> pd.Series:
        """
        Calculates the Commodity Channel Index (CCI).
        """
        typical_price = (self.high + self.low + self.close) / 3
        moving_average = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.subtract(moving_average).abs().rolling(window=period).mean()

        cci = (typical_price - moving_average) / (0.015 * mean_deviation)

        return cci

    def calculate_williams_r(self, period: int = 14) -> pd.Series:
        """
        Calculates the Williams %R.
        """
        highest_high = self.high.rolling(window=period).max()
        lowest_low = self.low.rolling(window=period).min()

        williams_r = (highest_high - self.close) / (highest_high - lowest_low) * -100

        return williams_r

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
        obv = pd.Series(index=self.close.index)
        obv.iloc[0] = 0

        for i in range(1, len(self.close)):
            if self.close[i] > self.close[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + self.volume[i]
            elif self.close[i] < self.close[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - self.volume[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    def calculate_cmf(self, period: int = 20) -> pd.Series:
        """
        Calculates the Chaikin Money Flow (CMF).
        """
        money_flow_multiplier = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)
        money_flow_volume = money_flow_multiplier * self.volume
        cmf = money_flow_volume.rolling(window=period).sum() / self.volume.rolling(window=period).sum()

        return cmf

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
    pass