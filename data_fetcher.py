"""
Data Fetcher for NSEApp v2
Unified data access layer with caching, wrapping 5paisa API
"""

import pandas as pd
import datetime as dt
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import sys
import time

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "options_strategies"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from cache_manager import CacheManager, CacheTTL, cached
from fno_stocks import NSE_FNO_STOCKS, NSE_INDICES, is_index


class DataFetcher:
    """
    Unified data fetcher for NSEApp v2
    Wraps 5paisa API with caching layer
    """
    
    def __init__(self, auto_authenticate: bool = True):
        """
        Initialize DataFetcher
        
        Args:
            auto_authenticate: Whether to authenticate with 5paisa on init
        """
        self.client = None
        self.auth = None
        self.instrument_df = None
        self._cache = CacheManager(default_ttl=CacheTTL.QUOTES)
        
        if auto_authenticate:
            self._authenticate()
    
    def _authenticate(self) -> bool:
        """Authenticate with 5paisa"""
        try:
            from auth_5paisa import get_auth_instance
            self.auth = get_auth_instance()
            self.client = self.auth.get_client()
            self.instrument_df = self.auth.load_scripmaster()
            print("✅ DataFetcher authenticated with 5paisa")
            return True
        except Exception as e:
            print(f"❌ DataFetcher authentication failed: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to 5paisa"""
        return self.client is not None and self.auth is not None
    
    # ========================================================================
    # SPOT PRICES & QUOTES
    # ========================================================================
    
    def get_spot_price(self, symbol: str) -> Optional[float]:
        """
        Get current spot price for a symbol (cached)
        
        Args:
            symbol: Stock/Index symbol
            
        Returns:
            Spot price or None if error
        """
        cache_key = f"spot:{symbol}"
        cached_price = self._cache.get(cache_key)
        if cached_price is not None:
            return cached_price
        
        try:
            scripcode = self.auth.get_scripcode(symbol)
            if not scripcode:
                return None
            
            response = self.client.fetch_market_depth([{
                "Exchange": "N",
                "ExchangeType": "C",
                "ScripCode": str(scripcode)
            }])
            
            price = response['Data'][0]['LastTradedPrice']
            self._cache.set(cache_key, price, CacheTTL.QUOTES)
            return price
            
        except Exception as e:
            print(f"Error fetching spot price for {symbol}: {e}")
            return None
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get full quote for a symbol (cached)
        
        Args:
            symbol: Stock/Index symbol
            
        Returns:
            Dictionary with price, change, volume, OI, etc.
        """
        cache_key = f"quote:{symbol}"
        cached_quote = self._cache.get(cache_key)
        if cached_quote is not None:
            return cached_quote
        
        try:
            scripcode = self.auth.get_scripcode(symbol)
            if not scripcode:
                return None
            
            response = self.client.fetch_market_depth([{
                "Exchange": "N",
                "ExchangeType": "C",
                "ScripCode": str(scripcode)
            }])
            
            data = response['Data'][0]
            
            quote = {
                "symbol": symbol,
                "ltp": data.get('LastTradedPrice', 0),
                "open": data.get('Open', 0),
                "high": data.get('High', 0),
                "low": data.get('Low', 0),
                "close": data.get('PClose', 0),
                "change": data.get('LastTradedPrice', 0) - data.get('PClose', 0),
                "change_pct": ((data.get('LastTradedPrice', 0) - data.get('PClose', 0)) / data.get('PClose', 1)) * 100 if data.get('PClose', 0) > 0 else 0,
                "volume": data.get('TotalQty', 0),
                "bid": data.get('BuyPrice', 0),
                "ask": data.get('SellPrice', 0),
                "bid_qty": data.get('BuyQty', 0),
                "ask_qty": data.get('SellQty', 0),
                "timestamp": dt.datetime.now().isoformat(),
            }
            
            self._cache.set(cache_key, quote, CacheTTL.QUOTES)
            return quote
            
        except Exception as e:
            print(f"Error fetching quote for {symbol}: {e}")
            return None
    
    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary mapping symbol to quote
        """
        results = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                results[symbol] = quote
        return results
    
    # ========================================================================
    # EXPIRY & OPTION CHAIN
    # ========================================================================
    
    def get_expiry_list(self, symbol: str) -> List[Dict]:
        """
        Get list of expiry dates for a symbol (cached)
        
        Args:
            symbol: Stock/Index symbol
            
        Returns:
            List of expiry dates with timestamp and formatted string
        """
        cache_key = f"expiry_list:{symbol}"
        cached_list = self._cache.get(cache_key)
        if cached_list is not None:
            return cached_list
        
        try:
            expiry_data = self.client.get_expiry("N", symbol)
            expiry_list = pd.DataFrame(expiry_data['Expiry'])
            
            def process_expiry(date_str):
                timestamp = int(date_str.split('(')[1].split('+')[0])
                date = dt.datetime.utcfromtimestamp(timestamp / 1000.0)
                return {
                    "timestamp": timestamp,
                    "date": date.strftime('%Y-%m-%d'),
                    "formatted": date.strftime('%d %b %Y'),
                    "days_to_expiry": (date.date() - dt.date.today()).days
                }
            
            expiries = [process_expiry(row['ExpiryDate']) for _, row in expiry_list.iterrows()]
            # Filter future expiries
            expiries = [e for e in expiries if e['days_to_expiry'] >= 0]
            
            self._cache.set(cache_key, expiries, CacheTTL.EXPIRY_LIST)
            return expiries
            
        except Exception as e:
            print(f"Error fetching expiry list for {symbol}: {e}")
            return []
    
    def get_next_expiry(self, symbol: str) -> Optional[Dict]:
        """
        Get nearest expiry for a symbol
        
        Args:
            symbol: Stock/Index symbol
            
        Returns:
            Dictionary with expiry details or None
        """
        expiries = self.get_expiry_list(symbol)
        if expiries:
            return expiries[0]
        return None
    
    def get_option_chain(self, symbol: str, expiry_timestamp: int) -> pd.DataFrame:
        """
        Fetch option chain for a symbol and expiry (cached)
        
        Args:
            symbol: Stock/Index symbol
            expiry_timestamp: Expiry timestamp
            
        Returns:
            DataFrame with option chain data
        """
        cache_key = f"chain:{symbol}:{expiry_timestamp}"
        cached_chain = self._cache.get(cache_key)
        if cached_chain is not None:
            return cached_chain
        
        try:
            option_chain = self.client.get_option_chain("N", symbol, expiry_timestamp)
            df = pd.DataFrame(option_chain['Options'])
            
            # Get spot price from expiry data
            expiry_data = self.client.get_expiry("N", symbol)
            spot_price = expiry_data['lastrate'][0]['LTP']
            df['SPOT'] = spot_price
            
            # Add some calculated columns
            if not df.empty:
                df['Moneyness'] = df.apply(
                    lambda row: 'ITM' if (row['CPType'] == 'CE' and row['StrikeRate'] < spot_price) or 
                                          (row['CPType'] == 'PE' and row['StrikeRate'] > spot_price)
                                else ('ATM' if row['StrikeRate'] == self._calculate_atm(spot_price, symbol) 
                                      else 'OTM'),
                    axis=1
                )
            
            self._cache.set(cache_key, df, CacheTTL.OPTION_CHAIN)
            return df
            
        except Exception as e:
            print(f"Error fetching option chain for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_full_chain_with_greeks(self, symbol: str, expiry_timestamp: int = None) -> pd.DataFrame:
        """
        Get option chain with Greeks calculated
        
        Args:
            symbol: Stock/Index symbol
            expiry_timestamp: Expiry timestamp (uses next expiry if not provided)
            
        Returns:
            DataFrame with option chain and Greeks
        """
        if expiry_timestamp is None:
            expiry = self.get_next_expiry(symbol)
            if not expiry:
                return pd.DataFrame()
            expiry_timestamp = expiry['timestamp']
        
        chain = self.get_option_chain(symbol, expiry_timestamp)
        if chain.empty:
            return chain
        
        # Calculate Greeks for each option
        spot = chain.iloc[0]['SPOT'] if not chain.empty else 0
        
        # Get days to expiry
        expiry = self.get_next_expiry(symbol)
        days_to_expiry = expiry['days_to_expiry'] if expiry else 1
        if days_to_expiry <= 0:
            days_to_expiry = 1
        
        greeks_data = []
        for _, row in chain.iterrows():
            greeks = self._calculate_greeks(
                spot=spot,
                strike=row['StrikeRate'],
                days=days_to_expiry,
                price=row['LastRate'],
                option_type=row['CPType']
            )
            greeks_data.append(greeks)
        
        greeks_df = pd.DataFrame(greeks_data)
        result = pd.concat([chain.reset_index(drop=True), greeks_df], axis=1)
        
        return result
    
    # ========================================================================
    # FUTURES DATA
    # ========================================================================
    
    def get_futures_quote(self, symbol: str, expiry_month: str = "current") -> Optional[Dict]:
        """
        Get futures quote for a symbol
        
        Args:
            symbol: Stock/Index symbol
            expiry_month: "current", "next", or "far"
            
        Returns:
            Futures quote dictionary
        """
        cache_key = f"futures:{symbol}:{expiry_month}"
        cached_quote = self._cache.get(cache_key)
        if cached_quote is not None:
            return cached_quote
        
        try:
            # Get futures from instrument_df
            if self.instrument_df is None:
                self.instrument_df = self.auth.load_scripmaster()
            
            futures = self.instrument_df[
                (self.instrument_df['Name'].str.contains(symbol, case=False)) &
                (self.instrument_df['Series'] == 'XX')  # Futures series
            ].copy()
            
            if futures.empty:
                return None
            
            # Sort by expiry and get requested month
            futures = futures.sort_values('Expiry Date')
            
            month_map = {"current": 0, "next": 1, "far": 2}
            idx = month_map.get(expiry_month, 0)
            
            if idx >= len(futures):
                idx = len(futures) - 1
            
            future_row = futures.iloc[idx]
            scripcode = int(future_row['Scripcode'])
            
            # Get quote
            response = self.client.fetch_market_depth([{
                "Exchange": "N",
                "ExchangeType": "D",
                "ScripCode": str(scripcode)
            }])
            
            data = response['Data'][0]
            
            quote = {
                "symbol": symbol,
                "scripcode": scripcode,
                "expiry": future_row['Expiry Date'],
                "ltp": data.get('LastTradedPrice', 0),
                "open": data.get('Open', 0),
                "high": data.get('High', 0),
                "low": data.get('Low', 0),
                "close": data.get('PClose', 0),
                "change": data.get('LastTradedPrice', 0) - data.get('PClose', 0),
                "change_pct": ((data.get('LastTradedPrice', 0) - data.get('PClose', 0)) / data.get('PClose', 1)) * 100 if data.get('PClose', 0) > 0 else 0,
                "oi": data.get('OpenInterest', 0),
                "oi_change": data.get('OIChange', 0),
                "volume": data.get('TotalQty', 0),
                "timestamp": dt.datetime.now().isoformat(),
            }
            
            self._cache.set(cache_key, quote, CacheTTL.QUOTES)
            return quote
            
        except Exception as e:
            print(f"Error fetching futures quote for {symbol}: {e}")
            return None
    
    # ========================================================================
    # HISTORICAL DATA
    # ========================================================================
    
    def get_historical_data(
        self, 
        symbol: str, 
        days: int = 30,
        interval: str = "1D"
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data
        
        Args:
            symbol: Stock/Index symbol
            days: Number of days of history
            interval: "1D" for daily, "1H" for hourly, etc.
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"history:{symbol}:{days}:{interval}"
        cached_data = self._cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            scripcode = self.auth.get_scripcode(symbol)
            if not scripcode:
                return pd.DataFrame()
            
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=days)
            
            # Map interval to 5paisa format
            interval_map = {
                "1D": "1d",
                "1H": "1h",
                "15M": "15m",
                "5M": "5m",
                "1M": "1m"
            }
            
            api_interval = interval_map.get(interval, "1d")
            
            response = self.client.historical_data(
                Exch="N",
                ExchangeSegment="C",
                ScripCode=scripcode,
                time=api_interval,
                From=start_date.strftime("%Y-%m-%d"),
                To=end_date.strftime("%Y-%m-%d")
            )
            
            if 'candleData' in response:
                df = pd.DataFrame(response['candleData'])
                df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime')
                
                self._cache.set(cache_key, df, CacheTTL.QUOTES)
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _calculate_atm(self, spot_price: float, symbol: str) -> int:
        """Calculate ATM strike"""
        strike_intervals = {
            'NIFTY': 50,
            'BANKNIFTY': 100,
            'FINNIFTY': 50,
            'MIDCPNIFTY': 25
        }
        
        interval = strike_intervals.get(symbol, 10)
        threshold = interval // 2
        
        mod = int(spot_price) % interval
        
        if mod < threshold:
            return int((spot_price // interval) * interval)
        else:
            return int(((spot_price // interval) + 1) * interval)
    
    def _calculate_greeks(
        self, 
        spot: float, 
        strike: float, 
        days: int, 
        price: float, 
        option_type: str,
        rate: float = 7.0
    ) -> Dict[str, float]:
        """Calculate option Greeks using Mibian"""
        try:
            import mibian as mb
            
            if days <= 0:
                days = 1
            if price <= 0:
                return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "iv": 0}
            
            if option_type == 'CE':
                bs = mb.BS([spot, strike, rate, days], callPrice=price)
                iv = bs.impliedVolatility
                bs_greeks = mb.BS([spot, strike, rate, days], volatility=iv)
                
                return {
                    "delta": round(bs_greeks.callDelta, 4),
                    "gamma": round(bs_greeks.gamma, 6),
                    "theta": round(bs_greeks.callTheta, 4),
                    "vega": round(bs_greeks.vega, 4),
                    "iv": round(iv, 2)
                }
            else:
                bs = mb.BS([spot, strike, rate, days], putPrice=price)
                iv = bs.impliedVolatility
                bs_greeks = mb.BS([spot, strike, rate, days], volatility=iv)
                
                return {
                    "delta": round(bs_greeks.putDelta, 4),
                    "gamma": round(bs_greeks.gamma, 6),
                    "theta": round(bs_greeks.putTheta, 4),
                    "vega": round(bs_greeks.vega, 4),
                    "iv": round(iv, 2)
                }
                
        except Exception as e:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "iv": 0}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self._cache.get_stats()
    
    def clear_cache(self) -> int:
        """Clear all cached data"""
        return self._cache.clear()


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_fetcher_instance: Optional[DataFetcher] = None


def get_data_fetcher() -> DataFetcher:
    """Get or create singleton DataFetcher instance"""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = DataFetcher()
    return _fetcher_instance


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing DataFetcher...")
    print("=" * 50)
    
    fetcher = DataFetcher()
    
    if fetcher.is_connected():
        print("\n1. Testing spot price:")
        spot = fetcher.get_spot_price("NIFTY")
        print(f"   NIFTY spot: ₹{spot}")
        
        print("\n2. Testing full quote:")
        quote = fetcher.get_quote("RELIANCE")
        if quote:
            print(f"   RELIANCE: ₹{quote['ltp']} ({quote['change_pct']:.2f}%)")
        
        print("\n3. Testing expiry list:")
        expiries = fetcher.get_expiry_list("NIFTY")
        if expiries:
            print(f"   Next 3 NIFTY expiries:")
            for exp in expiries[:3]:
                print(f"     {exp['formatted']} ({exp['days_to_expiry']} days)")
        
        print("\n4. Testing option chain:")
        next_exp = fetcher.get_next_expiry("NIFTY")
        if next_exp:
            chain = fetcher.get_option_chain("NIFTY", next_exp['timestamp'])
            print(f"   Option chain size: {len(chain)} options")
        
        print("\n5. Cache stats:")
        stats = fetcher.get_cache_stats()
        print(f"   {stats}")
        
        print("\n✅ DataFetcher test complete!")
    else:
        print("❌ DataFetcher not connected - skipping tests")
