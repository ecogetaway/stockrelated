"""
Futures OI Synopsis Module for NSEApp v2
Provides Futures Open Interest analysis with buildup classification
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.buildup_classifier import (
    BuildupType, classify_buildup, classify_with_details,
    classify_dataframe, BUILDUP_SHORT_NAMES, BUILDUP_COLORS
)
from core.fno_stocks import NSE_FNO_STOCKS, PRIORITY_SECTORS, get_stocks_by_sector
from config import DISPLAY_CONFIG


@dataclass
class FuturesOIData:
    """Represents futures OI data for a stock"""
    symbol: str
    ltp: float
    prev_close: float
    price_change: float
    price_change_pct: float
    oi: int
    prev_oi: int
    oi_change: int
    oi_change_pct: float
    volume: int
    buildup: str
    sentiment: str
    color: str


class FuturesOISynopsis:
    """
    Futures OI Synopsis Module
    Analyzes futures OI data and provides buildup classification
    """
    
    def __init__(self, data_fetcher=None):
        """
        Initialize FuturesOISynopsis
        
        Args:
            data_fetcher: Optional DataFetcher instance
        """
        self.data_fetcher = data_fetcher
        self._data_cache: Dict[str, FuturesOIData] = {}
    
    def set_data_fetcher(self, data_fetcher):
        """Set the data fetcher instance"""
        self.data_fetcher = data_fetcher
    
    def fetch_futures_data(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Fetch futures data for given symbols
        
        Args:
            symbols: List of symbols (defaults to all FNO stocks)
            
        Returns:
            DataFrame with futures data and buildup classification
        """
        if symbols is None:
            symbols = NSE_FNO_STOCKS
        
        if self.data_fetcher is None:
            print("⚠️ No data fetcher configured - using mock data")
            return self._get_mock_data(symbols)
        
        data_list = []
        
        for symbol in symbols:
            try:
                futures_quote = self.data_fetcher.get_futures_quote(symbol)
                
                if futures_quote:
                    data_list.append({
                        'symbol': symbol,
                        'ltp': futures_quote['ltp'],
                        'close': futures_quote['close'],
                        'oi': futures_quote['oi'],
                        'prev_oi': futures_quote['oi'] - futures_quote.get('oi_change', 0),
                        'volume': futures_quote['volume']
                    })
            except Exception as e:
                print(f"Error fetching futures data for {symbol}: {e}")
                continue
        
        if not data_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list)
        
        # Apply buildup classification
        classified_df = classify_dataframe(
            df, 
            price_col='ltp',
            prev_price_col='close',
            oi_col='oi',
            prev_oi_col='prev_oi',
            symbol_col='symbol'
        )
        
        return classified_df
    
    def _get_mock_data(self, symbols: List[str]) -> pd.DataFrame:
        """Generate mock data for testing/demo purposes"""
        import random
        
        data = []
        for symbol in symbols[:50]:  # Limit for demo
            prev_close = random.uniform(100, 5000)
            price_change_pct = random.uniform(-5, 5)
            ltp = prev_close * (1 + price_change_pct / 100)
            
            prev_oi = random.randint(100000, 10000000)
            oi_change_pct = random.uniform(-10, 10)
            oi = int(prev_oi * (1 + oi_change_pct / 100))
            
            data.append({
                'symbol': symbol,
                'ltp': round(ltp, 2),
                'close': round(prev_close, 2),
                'oi': oi,
                'prev_oi': prev_oi,
                'volume': random.randint(10000, 1000000)
            })
        
        df = pd.DataFrame(data)
        return classify_dataframe(df)
    
    def get_buildup_summary(self, df: pd.DataFrame = None) -> Dict[str, int]:
        """
        Get count of stocks in each buildup category
        
        Returns:
            Dictionary with buildup type counts
        """
        if df is None:
            df = self.fetch_futures_data()
        
        if df.empty or 'buildup' not in df.columns:
            return {bt.value: 0 for bt in BuildupType}
        
        counts = df['buildup'].value_counts().to_dict()
        result = {bt.value: 0 for bt in BuildupType}
        result.update(counts)
        
        return result
    
    def get_long_buildup(self, df: pd.DataFrame = None, top_n: int = 10) -> pd.DataFrame:
        """Get top stocks with Long buildup (price up + OI up)"""
        if df is None:
            df = self.fetch_futures_data()
        
        long_stocks = df[df['buildup'] == BuildupType.LONG.value]
        return long_stocks.nlargest(top_n, 'oi_change_pct')
    
    def get_short_buildup(self, df: pd.DataFrame = None, top_n: int = 10) -> pd.DataFrame:
        """Get top stocks with Short buildup (price down + OI up)"""
        if df is None:
            df = self.fetch_futures_data()
        
        short_stocks = df[df['buildup'] == BuildupType.SHORT.value]
        return short_stocks.nlargest(top_n, 'oi_change_pct')
    
    def get_long_unwinding(self, df: pd.DataFrame = None, top_n: int = 10) -> pd.DataFrame:
        """Get top stocks with Long Unwinding (price down + OI down)"""
        if df is None:
            df = self.fetch_futures_data()
        
        lu_stocks = df[df['buildup'] == BuildupType.LONG_UNWINDING.value]
        return lu_stocks.nsmallest(top_n, 'oi_change_pct')
    
    def get_short_covering(self, df: pd.DataFrame = None, top_n: int = 10) -> pd.DataFrame:
        """Get top stocks with Short Covering (price up + OI down)"""
        if df is None:
            df = self.fetch_futures_data()
        
        sc_stocks = df[df['buildup'] == BuildupType.SHORT_COVERING.value]
        return sc_stocks.nsmallest(top_n, 'oi_change_pct')
    
    def get_by_sector(
        self, 
        sector: str, 
        df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Get futures OI data filtered by sector
        
        Args:
            sector: Sector name
            df: Pre-fetched DataFrame (optional)
            
        Returns:
            Filtered DataFrame
        """
        if df is None:
            sector_stocks = get_stocks_by_sector(sector)
            df = self.fetch_futures_data(sector_stocks)
        else:
            sector_stocks = get_stocks_by_sector(sector)
            df = df[df['symbol'].isin(sector_stocks)]
        
        return df
    
    def get_priority_sectors_summary(self) -> Dict[str, Dict]:
        """
        Get summary for priority sectors
        
        Returns:
            Dictionary with sector name -> buildup summary
        """
        result = {}
        
        for sector in PRIORITY_SECTORS:
            sector_df = self.get_by_sector(sector)
            if not sector_df.empty:
                result[sector] = {
                    'count': len(sector_df),
                    'summary': self.get_buildup_summary(sector_df),
                    'top_long': sector_df[sector_df['buildup'] == 'Long'].head(3)[['symbol', 'price_change_pct', 'oi_change_pct']].to_dict('records'),
                    'top_short': sector_df[sector_df['buildup'] == 'Short'].head(3)[['symbol', 'price_change_pct', 'oi_change_pct']].to_dict('records'),
                }
        
        return result
    
    def format_for_display(
        self, 
        df: pd.DataFrame, 
        columns: List[str] = None
    ) -> List[Dict]:
        """
        Format data for Streamlit display
        
        Args:
            df: DataFrame to format
            columns: Columns to include
            
        Returns:
            List of dictionaries for display
        """
        if df.empty:
            return []
        
        if columns is None:
            columns = [
                'symbol', 'ltp', 'price_change_pct', 
                'oi', 'oi_change_pct', 'buildup', 
                'buildup_color', 'sentiment'
            ]
        
        available_cols = [c for c in columns if c in df.columns]
        
        return df[available_cols].to_dict('records')


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Futures OI Synopsis Test")
    print("=" * 50)
    
    synopsis = FuturesOISynopsis()
    
    # Get mock data
    print("\n1. Fetching futures data (mock):")
    df = synopsis.fetch_futures_data()
    print(f"   Fetched {len(df)} stocks")
    
    # Buildup summary
    print("\n2. Buildup Summary:")
    summary = synopsis.get_buildup_summary(df)
    for buildup_type, count in summary.items():
        print(f"   {buildup_type}: {count}")
    
    # Top buildups
    print("\n3. Top Long Buildup:")
    long_df = synopsis.get_long_buildup(df, top_n=5)
    if not long_df.empty:
        for _, row in long_df.iterrows():
            print(f"   {row['symbol']}: Price {row['price_change_pct']:+.2f}%, OI {row['oi_change_pct']:+.2f}%")
    
    print("\n4. Top Short Buildup:")
    short_df = synopsis.get_short_buildup(df, top_n=5)
    if not short_df.empty:
        for _, row in short_df.iterrows():
            print(f"   {row['symbol']}: Price {row['price_change_pct']:+.2f}%, OI {row['oi_change_pct']:+.2f}%")
    
    print("\n✅ Futures OI Synopsis test complete!")
