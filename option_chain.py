"""
Option Chain Analyzer for NSEApp v2
Comprehensive option chain analysis with Greeks, PCR, Max Pain, and OI Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.greeks import GreeksCalculator, GreeksResult
from config import DISPLAY_CONFIG


@dataclass
class MaxPainResult:
    """Max Pain calculation result"""
    max_pain_strike: float
    total_pain_at_max: float
    call_pain: float
    put_pain: float
    pain_by_strike: Dict[float, float]


@dataclass
class PCRResult:
    """Put-Call Ratio result"""
    pcr_oi: float
    pcr_volume: float
    total_call_oi: int
    total_put_oi: int
    total_call_volume: int
    total_put_volume: int
    interpretation: str


class OptionChainAnalyzer:
    """
    Comprehensive Option Chain Analyzer
    Provides full chain analysis with Greeks, PCR, Max Pain, OI analysis
    """
    
    def __init__(self, data_fetcher=None):
        """
        Initialize OptionChainAnalyzer
        
        Args:
            data_fetcher: Optional DataFetcher instance
        """
        self.data_fetcher = data_fetcher
        self.greeks_calc = GreeksCalculator()
        self.strikes_above_atm = DISPLAY_CONFIG.get('chain_strikes_above_atm', 15)
        self.strikes_below_atm = DISPLAY_CONFIG.get('chain_strikes_below_atm', 15)
    
    def set_data_fetcher(self, data_fetcher):
        """Set the data fetcher instance"""
        self.data_fetcher = data_fetcher
    
    # ========================================================================
    # OPTION CHAIN FETCHING
    # ========================================================================
    
    def get_option_chain(
        self, 
        symbol: str, 
        expiry_timestamp: int = None,
        include_greeks: bool = True
    ) -> pd.DataFrame:
        """
        Fetch and process option chain
        
        Args:
            symbol: Stock/Index symbol
            expiry_timestamp: Expiry timestamp (uses next expiry if None)
            include_greeks: Whether to calculate Greeks
            
        Returns:
            Processed DataFrame with option chain
        """
        if self.data_fetcher is None:
            return self._get_mock_chain(symbol)
        
        try:
            # Get expiry if not provided
            if expiry_timestamp is None:
                expiry = self.data_fetcher.get_next_expiry(symbol)
                if not expiry:
                    return pd.DataFrame()
                expiry_timestamp = expiry['timestamp']
                days_to_expiry = expiry['days_to_expiry']
            else:
                # Calculate days to expiry
                expiries = self.data_fetcher.get_expiry_list(symbol)
                exp_match = [e for e in expiries if e['timestamp'] == expiry_timestamp]
                days_to_expiry = exp_match[0]['days_to_expiry'] if exp_match else 7
            
            # Fetch raw chain
            chain = self.data_fetcher.get_option_chain(symbol, expiry_timestamp)
            
            if chain.empty:
                return chain
            
            spot = chain.iloc[0]['SPOT'] if 'SPOT' in chain.columns else 0
            
            # Calculate Greeks if requested
            if include_greeks:
                chain = self.greeks_calc.calculate_chain_greeks(
                    chain_df=chain,
                    spot=spot,
                    days_to_expiry=days_to_expiry
                )
            
            # Add additional columns
            chain = self._enrich_chain(chain, spot)
            
            return chain
            
        except Exception as e:
            print(f"Error fetching option chain for {symbol}: {e}")
            return pd.DataFrame()
    
    def _enrich_chain(self, chain: pd.DataFrame, spot: float) -> pd.DataFrame:
        """Add calculated columns to chain"""
        if chain.empty:
            return chain
        
        # Calculate ATM strike
        atm_strike = self._get_atm_strike(spot, chain)
        
        # Add moneyness
        chain['is_atm'] = chain['StrikeRate'] == atm_strike
        chain['is_itm'] = chain.apply(
            lambda r: (r['CPType'] == 'CE' and r['StrikeRate'] < spot) or
                      (r['CPType'] == 'PE' and r['StrikeRate'] > spot),
            axis=1
        )
        chain['is_otm'] = ~chain['is_itm'] & ~chain['is_atm']
        
        # Calculate distance from ATM
        chain['distance_from_atm'] = abs(chain['StrikeRate'] - spot)
        chain['distance_pct'] = ((chain['StrikeRate'] - spot) / spot * 100).round(2)
        
        # Bid-Ask spread (if available)
        if 'BuyPrice' in chain.columns and 'SellPrice' in chain.columns:
            chain['spread'] = chain['SellPrice'] - chain['BuyPrice']
            chain['spread_pct'] = (chain['spread'] / chain['LastRate'] * 100).round(2)
        
        return chain
    
    def _get_atm_strike(self, spot: float, chain: pd.DataFrame) -> float:
        """Find ATM strike from chain"""
        if chain.empty:
            return spot
        
        strikes = chain['StrikeRate'].unique()
        atm_strike = min(strikes, key=lambda x: abs(x - spot))
        return atm_strike
    
    def _get_mock_chain(self, symbol: str) -> pd.DataFrame:
        """Generate mock chain for testing"""
        import random
        
        # Determine spot based on symbol
        if symbol in ['NIFTY']:
            spot = 24000
            strike_interval = 50
        elif symbol in ['BANKNIFTY']:
            spot = 51000
            strike_interval = 100
        else:
            spot = random.uniform(500, 5000)
            strike_interval = 10
        
        atm = int(spot / strike_interval) * strike_interval
        strikes = [atm + i * strike_interval for i in range(-15, 16)]
        
        data = []
        for strike in strikes:
            for cp_type in ['CE', 'PE']:
                # Generate realistic option prices
                distance = abs(strike - spot)
                if cp_type == 'CE':
                    intrinsic = max(0, spot - strike)
                else:
                    intrinsic = max(0, strike - spot)
                
                time_value = max(5, 100 - distance * 0.1) * random.uniform(0.8, 1.2)
                price = intrinsic + time_value
                
                oi = random.randint(10000, 5000000)
                oi_change = random.randint(-100000, 100000)
                volume = random.randint(1000, 500000)
                
                data.append({
                    'Symbol': symbol,
                    'StrikeRate': strike,
                    'CPType': cp_type,
                    'LastRate': round(price, 2),
                    'OpenInterest': oi,
                    'OIChange': oi_change,
                    'Volume': volume,
                    'BuyPrice': round(price * 0.98, 2),
                    'SellPrice': round(price * 1.02, 2),
                    'SPOT': spot,
                })
        
        df = pd.DataFrame(data)
        
        # Calculate Greeks
        df = self.greeks_calc.calculate_chain_greeks(df, spot, 7)
        df = self._enrich_chain(df, spot)
        
        return df
    
    # ========================================================================
    # PCR (PUT-CALL RATIO)
    # ========================================================================
    
    def calculate_pcr(self, chain: pd.DataFrame) -> PCRResult:
        """
        Calculate Put-Call Ratio
        
        Args:
            chain: Option chain DataFrame
            
        Returns:
            PCRResult with PCR values and interpretation
        """
        if chain.empty:
            return PCRResult(0, 0, 0, 0, 0, 0, "No data")
        
        calls = chain[chain['CPType'] == 'CE']
        puts = chain[chain['CPType'] == 'PE']
        
        total_call_oi = calls['OpenInterest'].sum() if 'OpenInterest' in calls.columns else 0
        total_put_oi = puts['OpenInterest'].sum() if 'OpenInterest' in puts.columns else 0
        
        total_call_vol = calls['Volume'].sum() if 'Volume' in calls.columns else 0
        total_put_vol = puts['Volume'].sum() if 'Volume' in puts.columns else 0
        
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        pcr_volume = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        
        # Interpretation
        if pcr_oi > 1.2:
            interpretation = "Bullish (High Put writing indicates support)"
        elif pcr_oi < 0.8:
            interpretation = "Bearish (High Call writing indicates resistance)"
        else:
            interpretation = "Neutral (Balanced OI)"
        
        return PCRResult(
            pcr_oi=round(pcr_oi, 3),
            pcr_volume=round(pcr_volume, 3),
            total_call_oi=int(total_call_oi),
            total_put_oi=int(total_put_oi),
            total_call_volume=int(total_call_vol),
            total_put_volume=int(total_put_vol),
            interpretation=interpretation
        )
    
    # ========================================================================
    # MAX PAIN
    # ========================================================================
    
    def calculate_max_pain(self, chain: pd.DataFrame, spot: float = None) -> MaxPainResult:
        """
        Calculate Max Pain (strike where option writers lose minimum)
        
        Args:
            chain: Option chain DataFrame
            spot: Current spot price (optional, uses from chain if available)
            
        Returns:
            MaxPainResult with max pain details
        """
        if chain.empty:
            return MaxPainResult(0, 0, 0, 0, {})
        
        if spot is None:
            spot = chain['SPOT'].iloc[0] if 'SPOT' in chain.columns else 0
        
        strikes = sorted(chain['StrikeRate'].unique())
        pain_by_strike = {}
        
        for test_strike in strikes:
            call_pain = 0
            put_pain = 0
            
            # For each strike, calculate pain if spot expires at test_strike
            for strike in strikes:
                # Get OI for this strike
                call_oi = chain[(chain['StrikeRate'] == strike) & (chain['CPType'] == 'CE')]['OpenInterest'].sum()
                put_oi = chain[(chain['StrikeRate'] == strike) & (chain['CPType'] == 'PE')]['OpenInterest'].sum()
                
                # Call pain: max(0, test_strike - strike) * call_oi
                if test_strike > strike:
                    call_pain += (test_strike - strike) * call_oi
                
                # Put pain: max(0, strike - test_strike) * put_oi
                if strike > test_strike:
                    put_pain += (strike - test_strike) * put_oi
            
            pain_by_strike[strike] = call_pain + put_pain
        
        # Find max pain (minimum total pain)
        max_pain_strike = min(pain_by_strike, key=pain_by_strike.get)
        min_pain = pain_by_strike[max_pain_strike]
        
        # Calculate call and put pain at max pain strike
        call_pain_at_max = 0
        put_pain_at_max = 0
        
        for strike in strikes:
            call_oi = chain[(chain['StrikeRate'] == strike) & (chain['CPType'] == 'CE')]['OpenInterest'].sum()
            put_oi = chain[(chain['StrikeRate'] == strike) & (chain['CPType'] == 'PE')]['OpenInterest'].sum()
            
            if max_pain_strike > strike:
                call_pain_at_max += (max_pain_strike - strike) * call_oi
            if strike > max_pain_strike:
                put_pain_at_max += (strike - max_pain_strike) * put_oi
        
        return MaxPainResult(
            max_pain_strike=max_pain_strike,
            total_pain_at_max=min_pain,
            call_pain=call_pain_at_max,
            put_pain=put_pain_at_max,
            pain_by_strike=pain_by_strike
        )
    
    # ========================================================================
    # OI ANALYSIS
    # ========================================================================
    
    def get_oi_analysis(self, chain: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive OI analysis
        
        Args:
            chain: Option chain DataFrame
            
        Returns:
            Dictionary with OI analysis
        """
        if chain.empty:
            return {}
        
        spot = chain['SPOT'].iloc[0] if 'SPOT' in chain.columns else 0
        
        # Get highest OI strikes
        calls = chain[chain['CPType'] == 'CE'].copy()
        puts = chain[chain['CPType'] == 'PE'].copy()
        
        highest_call_oi = calls.nlargest(5, 'OpenInterest') if not calls.empty else pd.DataFrame()
        highest_put_oi = puts.nlargest(5, 'OpenInterest') if not puts.empty else pd.DataFrame()
        
        # Get highest OI change
        highest_call_oi_change = calls.nlargest(5, 'OIChange') if 'OIChange' in calls.columns else pd.DataFrame()
        highest_put_oi_change = puts.nlargest(5, 'OIChange') if 'OIChange' in puts.columns else pd.DataFrame()
        
        # Calculate resistance and support levels
        resistance_levels = highest_call_oi['StrikeRate'].tolist() if not highest_call_oi.empty else []
        support_levels = highest_put_oi['StrikeRate'].tolist() if not highest_put_oi.empty else []
        
        # Immediate resistance and support
        call_strikes_above = calls[calls['StrikeRate'] > spot].nlargest(3, 'OpenInterest')
        put_strikes_below = puts[puts['StrikeRate'] < spot].nlargest(3, 'OpenInterest')
        
        immediate_resistance = call_strikes_above['StrikeRate'].tolist() if not call_strikes_above.empty else []
        immediate_support = put_strikes_below['StrikeRate'].tolist() if not put_strikes_below.empty else []
        
        return {
            'spot': spot,
            'highest_call_oi_strikes': highest_call_oi[['StrikeRate', 'OpenInterest']].to_dict('records') if not highest_call_oi.empty else [],
            'highest_put_oi_strikes': highest_put_oi[['StrikeRate', 'OpenInterest']].to_dict('records') if not highest_put_oi.empty else [],
            'highest_call_oi_change': highest_call_oi_change[['StrikeRate', 'OIChange']].to_dict('records') if not highest_call_oi_change.empty else [],
            'highest_put_oi_change': highest_put_oi_change[['StrikeRate', 'OIChange']].to_dict('records') if not highest_put_oi_change.empty else [],
            'resistance_levels': resistance_levels,
            'support_levels': support_levels,
            'immediate_resistance': immediate_resistance,
            'immediate_support': immediate_support,
        }
    
    # ========================================================================
    # CHAIN FORMATTING
    # ========================================================================
    
    def format_chain_for_display(
        self, 
        chain: pd.DataFrame,
        include_greeks: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Format chain for side-by-side display (Calls | Strike | Puts)
        
        Args:
            chain: Raw option chain
            include_greeks: Include Greeks columns
            
        Returns:
            Tuple of (calls_df, puts_df) formatted for display
        """
        if chain.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        calls = chain[chain['CPType'] == 'CE'].copy()
        puts = chain[chain['CPType'] == 'PE'].copy()
        
        # Common columns
        base_cols = ['StrikeRate', 'LastRate', 'OpenInterest', 'OIChange', 'Volume']
        greek_cols = ['delta', 'gamma', 'theta', 'vega', 'iv'] if include_greeks else []
        
        call_cols = [c for c in base_cols + greek_cols if c in calls.columns]
        put_cols = [c for c in base_cols + greek_cols if c in puts.columns]
        
        calls_formatted = calls[call_cols].sort_values('StrikeRate')
        puts_formatted = puts[put_cols].sort_values('StrikeRate')
        
        return calls_formatted, puts_formatted
    
    def get_straddle_strangle_data(
        self, 
        chain: pd.DataFrame, 
        spot: float = None
    ) -> Dict[str, Any]:
        """
        Get straddle and strangle premium data
        
        Args:
            chain: Option chain DataFrame
            spot: Current spot price
            
        Returns:
            Dictionary with straddle/strangle data
        """
        if chain.empty:
            return {}
        
        if spot is None:
            spot = chain['SPOT'].iloc[0] if 'SPOT' in chain.columns else 0
        
        atm_strike = self._get_atm_strike(spot, chain)
        
        # ATM Straddle
        atm_call = chain[(chain['StrikeRate'] == atm_strike) & (chain['CPType'] == 'CE')]
        atm_put = chain[(chain['StrikeRate'] == atm_strike) & (chain['CPType'] == 'PE')]
        
        straddle_premium = 0
        if not atm_call.empty and not atm_put.empty:
            straddle_premium = atm_call['LastRate'].iloc[0] + atm_put['LastRate'].iloc[0]
        
        # OTM Strangle (5% OTM each side)
        call_strike = atm_strike * 1.02  # 2% OTM call
        put_strike = atm_strike * 0.98   # 2% OTM put
        
        strikes = chain['StrikeRate'].unique()
        strangle_call_strike = min(strikes, key=lambda x: abs(x - call_strike) if x > atm_strike else float('inf'))
        strangle_put_strike = min(strikes, key=lambda x: abs(x - put_strike) if x < atm_strike else float('inf'))
        
        strangle_call = chain[(chain['StrikeRate'] == strangle_call_strike) & (chain['CPType'] == 'CE')]
        strangle_put = chain[(chain['StrikeRate'] == strangle_put_strike) & (chain['CPType'] == 'PE')]
        
        strangle_premium = 0
        if not strangle_call.empty and not strangle_put.empty:
            strangle_premium = strangle_call['LastRate'].iloc[0] + strangle_put['LastRate'].iloc[0]
        
        return {
            'atm_strike': atm_strike,
            'spot': spot,
            'straddle_premium': round(straddle_premium, 2),
            'straddle_breakeven_upper': round(atm_strike + straddle_premium, 2),
            'straddle_breakeven_lower': round(atm_strike - straddle_premium, 2),
            'strangle_call_strike': strangle_call_strike,
            'strangle_put_strike': strangle_put_strike,
            'strangle_premium': round(strangle_premium, 2),
            'strangle_breakeven_upper': round(strangle_call_strike + strangle_premium, 2),
            'strangle_breakeven_lower': round(strangle_put_strike - strangle_premium, 2),
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Option Chain Analyzer Test")
    print("=" * 50)
    
    analyzer = OptionChainAnalyzer()
    
    print("\n1. Fetching option chain (mock):")
    chain = analyzer.get_option_chain("NIFTY")
    print(f"   Fetched {len(chain)} options")
    print(f"   Columns: {list(chain.columns)}")
    
    print("\n2. PCR Analysis:")
    pcr = analyzer.calculate_pcr(chain)
    print(f"   PCR (OI): {pcr.pcr_oi}")
    print(f"   Total Call OI: {pcr.total_call_oi:,}")
    print(f"   Total Put OI: {pcr.total_put_oi:,}")
    print(f"   Interpretation: {pcr.interpretation}")
    
    print("\n3. Max Pain Analysis:")
    max_pain = analyzer.calculate_max_pain(chain)
    print(f"   Max Pain Strike: {max_pain.max_pain_strike}")
    print(f"   Total Pain at Max Pain: {max_pain.total_pain_at_max:,.0f}")
    
    print("\n4. OI Analysis:")
    oi_analysis = analyzer.get_oi_analysis(chain)
    print(f"   Spot: {oi_analysis['spot']}")
    print(f"   Immediate Resistance: {oi_analysis['immediate_resistance'][:3]}")
    print(f"   Immediate Support: {oi_analysis['immediate_support'][:3]}")
    
    print("\n5. Straddle/Strangle Data:")
    straddle = analyzer.get_straddle_strangle_data(chain)
    print(f"   ATM Strike: {straddle['atm_strike']}")
    print(f"   Straddle Premium: {straddle['straddle_premium']}")
    print(f"   Straddle Breakeven: {straddle['straddle_breakeven_lower']} - {straddle['straddle_breakeven_upper']}")
    
    print("\n6. Chain Format for Display:")
    calls_df, puts_df = analyzer.format_chain_for_display(chain)
    print(f"   Calls: {len(calls_df)} strikes")
    print(f"   Puts: {len(puts_df)} strikes")
    
    print("\n✅ Option Chain Analyzer test complete!")
