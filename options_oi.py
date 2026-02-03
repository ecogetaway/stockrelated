"""
Options OI Synopsis Module for NSEApp v2
Provides Options Open Interest analysis with Long Call, Long Put, Short Call, Short Put classification
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.buildup_classifier import BuildupType, classify_buildup, BUILDUP_COLORS
from core.fno_stocks import NSE_FNO_STOCKS, NSE_INDICES


class OptionsBuildup:
    """Options buildup types"""
    LONG_CALL = "Long Call"      # Call writers adding positions (bearish for underlying)
    LONG_PUT = "Long Put"        # Put buyers adding positions (bearish protection)
    SHORT_CALL = "Short Call"    # Call sellers unwinding (bullish)
    SHORT_PUT = "Short Put"      # Put sellers unwinding (bullish)


OPTIONS_BUILDUP_COLORS = {
    "Long Call": "#4CAF50",   # Green
    "Long Put": "#F44336",    # Red
    "Short Call": "#81D4FA",  # Light Blue
    "Short Put": "#F8BBD9",   # Pink
}


@dataclass
class OptionsOIData:
    """Represents options OI data for a stock/strike"""
    symbol: str
    strike: float
    option_type: str  # CE or PE
    ltp: float
    prev_close: float
    price_change_pct: float
    oi: int
    prev_oi: int
    oi_change: int
    oi_change_pct: float
    volume: int
    buildup: str
    iv: float


class OptionsOISynopsis:
    """
    Options OI Synopsis Module
    Analyzes options OI data and classifies buildup patterns
    """
    
    def __init__(self, data_fetcher=None):
        """
        Initialize OptionsOISynopsis
        
        Args:
            data_fetcher: Optional DataFetcher instance
        """
        self.data_fetcher = data_fetcher
    
    def set_data_fetcher(self, data_fetcher):
        """Set the data fetcher instance"""
        self.data_fetcher = data_fetcher
    
    def fetch_options_oi_data(self, symbol: str, expiry_timestamp: int = None) -> pd.DataFrame:
        """
        Fetch options OI data for a symbol
        
        Args:
            symbol: Stock/Index symbol
            expiry_timestamp: Expiry timestamp (uses next expiry if None)
            
        Returns:
            DataFrame with options OI data
        """
        if self.data_fetcher is None:
            return self._get_mock_data(symbol)
        
        try:
            if expiry_timestamp is None:
                expiry = self.data_fetcher.get_next_expiry(symbol)
                if not expiry:
                    return pd.DataFrame()
                expiry_timestamp = expiry['timestamp']
            
            chain = self.data_fetcher.get_option_chain(symbol, expiry_timestamp)
            
            if chain.empty:
                return pd.DataFrame()
            
            # Add buildup classification for each option
            chain['oi_change'] = chain.get('OIChange', 0)
            chain['oi_change_pct'] = (chain['oi_change'] / chain['OpenInterest'].replace(0, 1) * 100).round(2)
            chain['price_change'] = chain['LastRate'] - chain.get('PrevClose', chain['LastRate'])
            chain['price_change_pct'] = (chain['price_change'] / chain.get('PrevClose', chain['LastRate']).replace(0, 1) * 100).round(2)
            
            # Classify buildup
            chain['buildup'] = chain.apply(
                lambda row: self._classify_option_buildup(
                    row['price_change_pct'], 
                    row['oi_change_pct'],
                    row['CPType']
                ),
                axis=1
            )
            
            return chain
            
        except Exception as e:
            print(f"Error fetching options OI for {symbol}: {e}")
            return pd.DataFrame()
    
    def _classify_option_buildup(
        self, 
        price_change_pct: float, 
        oi_change_pct: float,
        option_type: str
    ) -> str:
        """
        Classify option buildup
        
        For Calls:
        - Price up + OI up = Long Call (buying calls, bullish)
        - Price down + OI down = Short Call exit (bearish)
        - Price down + OI up = Short Call buildup (bearish, writers adding)
        - Price up + OI down = Long Call exit (profit booking)
        
        For Puts:
        - Price up + OI up = Long Put (buying protection, bearish)
        - Price down + OI down = Short Put exit (bullish)
        - Price down + OI up = Short Put buildup (sellers adding, bullish)
        - Price up + OI down = Long Put exit (profit booking)
        """
        price_up = price_change_pct > 0
        oi_up = oi_change_pct > 0
        
        if option_type == 'CE':
            if price_up and oi_up:
                return "Long Call"
            elif not price_up and oi_up:
                return "Short Call"
            elif price_up and not oi_up:
                return "Long Call Exit"
            else:
                return "Short Call Exit"
        else:  # PE
            if price_up and oi_up:
                return "Long Put"
            elif not price_up and oi_up:
                return "Short Put"
            elif price_up and not oi_up:
                return "Long Put Exit"
            else:
                return "Short Put Exit"
    
    def _get_mock_data(self, symbol: str) -> pd.DataFrame:
        """Generate mock data for testing"""
        import random
        
        spot = random.uniform(15000, 25000) if symbol in NSE_INDICES else random.uniform(500, 5000)
        strikes = [int(spot * (1 + i * 0.02)) for i in range(-10, 11)]
        
        data = []
        for strike in strikes:
            for cp_type in ['CE', 'PE']:
                oi = random.randint(10000, 1000000)
                oi_change = random.randint(-50000, 50000)
                price = random.uniform(5, 200)
                price_change = random.uniform(-20, 20)
                
                data.append({
                    'symbol': symbol,
                    'StrikeRate': strike,
                    'CPType': cp_type,
                    'LastRate': round(price, 2),
                    'OpenInterest': oi,
                    'oi_change': oi_change,
                    'oi_change_pct': round(oi_change / oi * 100, 2),
                    'price_change_pct': round(price_change / price * 100, 2),
                    'Volume': random.randint(1000, 100000),
                    'IV': random.uniform(10, 50)
                })
        
        df = pd.DataFrame(data)
        df['buildup'] = df.apply(
            lambda row: self._classify_option_buildup(
                row['price_change_pct'], 
                row['oi_change_pct'],
                row['CPType']
            ),
            axis=1
        )
        
        return df
    
    def get_top_oi_gainers(self, df: pd.DataFrame = None, symbol: str = "NIFTY", top_n: int = 10) -> pd.DataFrame:
        """Get options with highest OI increase"""
        if df is None:
            df = self.fetch_options_oi_data(symbol)
        
        if df.empty:
            return df
        
        return df.nlargest(top_n, 'oi_change')
    
    def get_top_oi_losers(self, df: pd.DataFrame = None, symbol: str = "NIFTY", top_n: int = 10) -> pd.DataFrame:
        """Get options with highest OI decrease"""
        if df is None:
            df = self.fetch_options_oi_data(symbol)
        
        if df.empty:
            return df
        
        return df.nsmallest(top_n, 'oi_change')
    
    def get_call_put_summary(self, df: pd.DataFrame = None, symbol: str = "NIFTY") -> Dict:
        """
        Get summary of call vs put OI
        
        Returns:
            Dictionary with call/put OI summary
        """
        if df is None:
            df = self.fetch_options_oi_data(symbol)
        
        if df.empty:
            return {}
        
        calls = df[df['CPType'] == 'CE']
        puts = df[df['CPType'] == 'PE']
        
        return {
            'total_call_oi': calls['OpenInterest'].sum() if 'OpenInterest' in calls.columns else 0,
            'total_put_oi': puts['OpenInterest'].sum() if 'OpenInterest' in puts.columns else 0,
            'call_oi_change': calls['oi_change'].sum() if 'oi_change' in calls.columns else 0,
            'put_oi_change': puts['oi_change'].sum() if 'oi_change' in puts.columns else 0,
            'pcr': puts['OpenInterest'].sum() / max(calls['OpenInterest'].sum(), 1) if 'OpenInterest' in df.columns else 0,
        }
    
    def get_buildup_by_type(self, df: pd.DataFrame, buildup_type: str) -> pd.DataFrame:
        """Filter options by buildup type"""
        return df[df['buildup'] == buildup_type] if 'buildup' in df.columns else pd.DataFrame()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Options OI Synopsis Test")
    print("=" * 50)
    
    synopsis = OptionsOISynopsis()
    
    print("\n1. Fetching options OI data (mock):")
    df = synopsis.fetch_options_oi_data("NIFTY")
    print(f"   Fetched {len(df)} options")
    
    print("\n2. Call/Put Summary:")
    summary = synopsis.get_call_put_summary(df)
    for key, value in summary.items():
        print(f"   {key}: {value:,.0f}" if isinstance(value, (int, float)) else f"   {key}: {value}")
    
    print("\n3. Top OI Gainers:")
    gainers = synopsis.get_top_oi_gainers(df, top_n=5)
    for _, row in gainers.iterrows():
        print(f"   {row['StrikeRate']} {row['CPType']}: OI Change {row['oi_change']:+,}")
    
    print("\n✅ Options OI Synopsis test complete!")
