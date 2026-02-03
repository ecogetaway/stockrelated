"""
Movers Module for NSEApp v2
Provides Top Price Movers and OI Movers for the dashboard
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fno_stocks import NSE_FNO_STOCKS, PRIORITY_SECTORS, get_stocks_by_sector
from config import DISPLAY_CONFIG


@dataclass
class MoverData:
    """Represents a mover entry"""
    symbol: str
    ltp: float
    change: float
    change_pct: float
    volume: int
    oi: int = 0
    oi_change: int = 0
    oi_change_pct: float = 0.0


class Movers:
    """
    Movers Module
    Tracks top gainers, losers, volume leaders, and OI movers
    """
    
    def __init__(self, data_fetcher=None):
        """
        Initialize Movers
        
        Args:
            data_fetcher: Optional DataFetcher instance
        """
        self.data_fetcher = data_fetcher
        self.top_n = DISPLAY_CONFIG.get('top_movers_count', 10)
    
    def set_data_fetcher(self, data_fetcher):
        """Set the data fetcher instance"""
        self.data_fetcher = data_fetcher
    
    def fetch_all_quotes(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Fetch quotes for all symbols
        
        Args:
            symbols: List of symbols (defaults to all FNO stocks)
            
        Returns:
            DataFrame with quote data
        """
        if symbols is None:
            symbols = NSE_FNO_STOCKS
        
        if self.data_fetcher is None:
            return self._get_mock_data(symbols)
        
        data_list = []
        
        for symbol in symbols:
            try:
                quote = self.data_fetcher.get_quote(symbol)
                futures = self.data_fetcher.get_futures_quote(symbol)
                
                if quote:
                    data_list.append({
                        'symbol': symbol,
                        'ltp': quote['ltp'],
                        'change': quote['change'],
                        'change_pct': quote['change_pct'],
                        'volume': quote['volume'],
                        'oi': futures['oi'] if futures else 0,
                        'oi_change': futures.get('oi_change', 0) if futures else 0,
                    })
            except Exception as e:
                continue
        
        if not data_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list)
        df['oi_change_pct'] = (df['oi_change'] / df['oi'].replace(0, 1) * 100).round(2)
        
        return df
    
    def _get_mock_data(self, symbols: List[str]) -> pd.DataFrame:
        """Generate mock data for testing"""
        import random
        
        data = []
        for symbol in symbols[:100]:  # Limit for demo
            ltp = random.uniform(100, 5000)
            change_pct = random.uniform(-8, 8)
            change = ltp * change_pct / 100
            
            oi = random.randint(100000, 10000000)
            oi_change = random.randint(-500000, 500000)
            
            data.append({
                'symbol': symbol,
                'ltp': round(ltp, 2),
                'change': round(change, 2),
                'change_pct': round(change_pct, 2),
                'volume': random.randint(10000, 10000000),
                'oi': oi,
                'oi_change': oi_change,
                'oi_change_pct': round(oi_change / oi * 100, 2)
            })
        
        return pd.DataFrame(data)
    
    # ========================================================================
    # PRICE MOVERS
    # ========================================================================
    
    def get_top_gainers(self, df: pd.DataFrame = None, top_n: int = None) -> pd.DataFrame:
        """Get top price gainers"""
        if df is None:
            df = self.fetch_all_quotes()
        
        top_n = top_n or self.top_n
        return df.nlargest(top_n, 'change_pct')
    
    def get_top_losers(self, df: pd.DataFrame = None, top_n: int = None) -> pd.DataFrame:
        """Get top price losers"""
        if df is None:
            df = self.fetch_all_quotes()
        
        top_n = top_n or self.top_n
        return df.nsmallest(top_n, 'change_pct')
    
    def get_price_movers(self, df: pd.DataFrame = None, top_n: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get both gainers and losers"""
        if df is None:
            df = self.fetch_all_quotes()
        
        gainers = self.get_top_gainers(df, top_n)
        losers = self.get_top_losers(df, top_n)
        
        return gainers, losers
    
    # ========================================================================
    # VOLUME MOVERS
    # ========================================================================
    
    def get_volume_leaders(self, df: pd.DataFrame = None, top_n: int = None) -> pd.DataFrame:
        """Get stocks with highest volume"""
        if df is None:
            df = self.fetch_all_quotes()
        
        top_n = top_n or self.top_n
        return df.nlargest(top_n, 'volume')
    
    # ========================================================================
    # OI MOVERS
    # ========================================================================
    
    def get_oi_gainers(self, df: pd.DataFrame = None, top_n: int = None) -> pd.DataFrame:
        """Get stocks with highest OI increase"""
        if df is None:
            df = self.fetch_all_quotes()
        
        top_n = top_n or self.top_n
        return df.nlargest(top_n, 'oi_change')
    
    def get_oi_losers(self, df: pd.DataFrame = None, top_n: int = None) -> pd.DataFrame:
        """Get stocks with highest OI decrease"""
        if df is None:
            df = self.fetch_all_quotes()
        
        top_n = top_n or self.top_n
        return df.nsmallest(top_n, 'oi_change')
    
    def get_oi_movers(self, df: pd.DataFrame = None, top_n: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get both OI gainers and losers"""
        if df is None:
            df = self.fetch_all_quotes()
        
        gainers = self.get_oi_gainers(df, top_n)
        losers = self.get_oi_losers(df, top_n)
        
        return gainers, losers
    
    # ========================================================================
    # SECTOR MOVERS
    # ========================================================================
    
    def get_sector_movers(self, sector: str, df: pd.DataFrame = None, top_n: int = 5) -> Dict:
        """
        Get movers for a specific sector
        
        Args:
            sector: Sector name
            df: Pre-fetched data (optional)
            top_n: Number of movers per category
            
        Returns:
            Dictionary with gainers, losers, volume leaders
        """
        sector_stocks = get_stocks_by_sector(sector)
        
        if df is None:
            df = self.fetch_all_quotes(sector_stocks)
        else:
            df = df[df['symbol'].isin(sector_stocks)]
        
        if df.empty:
            return {'gainers': [], 'losers': [], 'volume': []}
        
        return {
            'gainers': df.nlargest(top_n, 'change_pct').to_dict('records'),
            'losers': df.nsmallest(top_n, 'change_pct').to_dict('records'),
            'volume': df.nlargest(top_n, 'volume').to_dict('records'),
        }
    
    def get_priority_sectors_movers(self, df: pd.DataFrame = None) -> Dict[str, Dict]:
        """Get movers for all priority sectors"""
        result = {}
        
        if df is None:
            df = self.fetch_all_quotes()
        
        for sector in PRIORITY_SECTORS:
            result[sector] = self.get_sector_movers(sector, df)
        
        return result
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    def get_market_summary(self, df: pd.DataFrame = None) -> Dict:
        """
        Get overall market summary
        
        Returns:
            Dictionary with market statistics
        """
        if df is None:
            df = self.fetch_all_quotes()
        
        if df.empty:
            return {}
        
        advancing = len(df[df['change_pct'] > 0])
        declining = len(df[df['change_pct'] < 0])
        unchanged = len(df[df['change_pct'] == 0])
        
        oi_up = len(df[df['oi_change'] > 0])
        oi_down = len(df[df['oi_change'] < 0])
        
        return {
            'total_stocks': len(df),
            'advancing': advancing,
            'declining': declining,
            'unchanged': unchanged,
            'advance_decline_ratio': round(advancing / max(declining, 1), 2),
            'oi_buildup': oi_up,
            'oi_reduction': oi_down,
            'avg_change_pct': round(df['change_pct'].mean(), 2),
            'total_volume': df['volume'].sum(),
        }
    
    def format_for_display(self, df: pd.DataFrame, display_type: str = 'price') -> List[Dict]:
        """Format data for Streamlit display"""
        if df.empty:
            return []
        
        if display_type == 'price':
            cols = ['symbol', 'ltp', 'change', 'change_pct']
        elif display_type == 'oi':
            cols = ['symbol', 'ltp', 'oi', 'oi_change', 'oi_change_pct']
        elif display_type == 'volume':
            cols = ['symbol', 'ltp', 'volume', 'change_pct']
        else:
            cols = df.columns.tolist()
        
        available_cols = [c for c in cols if c in df.columns]
        return df[available_cols].to_dict('records')


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Movers Module Test")
    print("=" * 50)
    
    movers = Movers()
    
    print("\n1. Fetching quotes (mock):")
    df = movers.fetch_all_quotes()
    print(f"   Fetched {len(df)} stocks")
    
    print("\n2. Top Gainers:")
    gainers = movers.get_top_gainers(df, top_n=5)
    for _, row in gainers.iterrows():
        print(f"   {row['symbol']}: {row['change_pct']:+.2f}%")
    
    print("\n3. Top Losers:")
    losers = movers.get_top_losers(df, top_n=5)
    for _, row in losers.iterrows():
        print(f"   {row['symbol']}: {row['change_pct']:+.2f}%")
    
    print("\n4. OI Gainers:")
    oi_gainers = movers.get_oi_gainers(df, top_n=5)
    for _, row in oi_gainers.iterrows():
        print(f"   {row['symbol']}: OI Change {row['oi_change']:+,}")
    
    print("\n5. Market Summary:")
    summary = movers.get_market_summary(df)
    print(f"   Advancing: {summary['advancing']}")
    print(f"   Declining: {summary['declining']}")
    print(f"   A/D Ratio: {summary['advance_decline_ratio']}")
    
    print("\n✅ Movers module test complete!")
