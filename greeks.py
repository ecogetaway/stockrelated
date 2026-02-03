"""
Greeks Calculator for NSEApp v2
Extended Greeks calculation with IV, historical volatility, and vol skew
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, date

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GREEKS_CONFIG

# Try to import mibian, provide fallback if not available
try:
    import mibian as mb
    MIBIAN_AVAILABLE = True
except ImportError:
    MIBIAN_AVAILABLE = False
    print("⚠️ mibian not installed. Using Black-Scholes fallback.")


@dataclass
class GreeksResult:
    """Result of Greeks calculation"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    iv: float  # Implied Volatility
    intrinsic_value: float
    time_value: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            "iv": self.iv,
            "intrinsic_value": self.intrinsic_value,
            "time_value": self.time_value,
        }


class GreeksCalculator:
    """
    Advanced Greeks Calculator
    Extends basic Greeks with additional analytics
    """
    
    def __init__(
        self, 
        risk_free_rate: float = None,
        dividend_yield: float = None,
        days_per_year: int = None
    ):
        """
        Initialize GreeksCalculator
        
        Args:
            risk_free_rate: Risk-free rate (default from config: 7%)
            dividend_yield: Dividend yield (default: 0%)
            days_per_year: Days per year for calculations (default: 365)
        """
        self.risk_free_rate = risk_free_rate or GREEKS_CONFIG.get("risk_free_rate", 7.0)
        self.dividend_yield = dividend_yield or GREEKS_CONFIG.get("dividend_yield", 0.0)
        self.days_per_year = days_per_year or GREEKS_CONFIG.get("days_per_year", 365)
    
    # ========================================================================
    # CORE GREEKS CALCULATION
    # ========================================================================
    
    def calculate_greeks(
        self,
        spot: float,
        strike: float,
        days_to_expiry: int,
        option_price: float,
        option_type: str = "CE",
        volatility: float = None
    ) -> GreeksResult:
        """
        Calculate all Greeks for an option
        
        Args:
            spot: Current spot price
            strike: Strike price
            days_to_expiry: Days until expiry
            option_price: Current option price (LTP)
            option_type: "CE" for Call, "PE" for Put
            volatility: Known volatility (if None, calculates IV from price)
            
        Returns:
            GreeksResult with all Greeks
        """
        if days_to_expiry <= 0:
            days_to_expiry = 1
        
        # Calculate intrinsic and time value
        if option_type == "CE":
            intrinsic = max(0, spot - strike)
        else:
            intrinsic = max(0, strike - spot)
        
        time_value = max(0, option_price - intrinsic)
        
        # Skip calculation if option price is zero or negligible
        if option_price <= 0.01:
            return GreeksResult(
                delta=0, gamma=0, theta=0, vega=0, rho=0,
                iv=0, intrinsic_value=intrinsic, time_value=0
            )
        
        try:
            if MIBIAN_AVAILABLE:
                return self._calculate_with_mibian(
                    spot, strike, days_to_expiry, option_price, 
                    option_type, volatility, intrinsic, time_value
                )
            else:
                return self._calculate_with_bs(
                    spot, strike, days_to_expiry, option_price,
                    option_type, volatility, intrinsic, time_value
                )
        except Exception as e:
            # Return zeros on calculation error
            return GreeksResult(
                delta=0, gamma=0, theta=0, vega=0, rho=0,
                iv=0, intrinsic_value=intrinsic, time_value=time_value
            )
    
    def _calculate_with_mibian(
        self, spot, strike, days, price, opt_type, vol, intrinsic, time_val
    ) -> GreeksResult:
        """Calculate Greeks using mibian library"""
        rate = self.risk_free_rate
        
        # Calculate IV if not provided
        if vol is None:
            if opt_type == "CE":
                bs_iv = mb.BS([spot, strike, rate, days], callPrice=price)
            else:
                bs_iv = mb.BS([spot, strike, rate, days], putPrice=price)
            iv = bs_iv.impliedVolatility
        else:
            iv = vol
        
        # Ensure IV is valid
        if iv is None or iv <= 0 or iv > 500:
            iv = 20  # Default to 20% if IV calculation fails
        
        # Calculate Greeks with IV
        bs_greeks = mb.BS([spot, strike, rate, days], volatility=iv)
        
        if opt_type == "CE":
            delta = bs_greeks.callDelta
            theta = bs_greeks.callTheta
            rho = bs_greeks.callRho if hasattr(bs_greeks, 'callRho') else 0
        else:
            delta = bs_greeks.putDelta
            theta = bs_greeks.putTheta
            rho = bs_greeks.putRho if hasattr(bs_greeks, 'putRho') else 0
        
        return GreeksResult(
            delta=round(delta, 4),
            gamma=round(bs_greeks.gamma, 6),
            theta=round(theta, 4),
            vega=round(bs_greeks.vega, 4),
            rho=round(rho, 4) if rho else 0,
            iv=round(iv, 2),
            intrinsic_value=round(intrinsic, 2),
            time_value=round(time_val, 2)
        )
    
    def _calculate_with_bs(
        self, spot, strike, days, price, opt_type, vol, intrinsic, time_val
    ) -> GreeksResult:
        """
        Calculate Greeks using Black-Scholes formula (fallback)
        """
        from scipy.stats import norm
        
        T = days / self.days_per_year
        r = self.risk_free_rate / 100
        q = self.dividend_yield / 100
        
        # Calculate IV using Newton-Raphson if not provided
        if vol is None:
            iv = self._calculate_iv_newton(spot, strike, T, r, q, price, opt_type)
        else:
            iv = vol / 100  # Convert to decimal
        
        sigma = iv
        
        if sigma <= 0:
            sigma = 0.2  # Default 20%
        
        # Calculate d1 and d2
        d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        # Calculate Greeks
        if opt_type == "CE":
            delta = math.exp(-q * T) * norm.cdf(d1)
            theta = (-spot * sigma * math.exp(-q * T) * norm.pdf(d1) / (2 * math.sqrt(T)) 
                    - r * strike * math.exp(-r * T) * norm.cdf(d2) 
                    + q * spot * math.exp(-q * T) * norm.cdf(d1)) / self.days_per_year
            rho = strike * T * math.exp(-r * T) * norm.cdf(d2) / 100
        else:
            delta = math.exp(-q * T) * (norm.cdf(d1) - 1)
            theta = (-spot * sigma * math.exp(-q * T) * norm.pdf(d1) / (2 * math.sqrt(T))
                    + r * strike * math.exp(-r * T) * norm.cdf(-d2)
                    - q * spot * math.exp(-q * T) * norm.cdf(-d1)) / self.days_per_year
            rho = -strike * T * math.exp(-r * T) * norm.cdf(-d2) / 100
        
        gamma = math.exp(-q * T) * norm.pdf(d1) / (spot * sigma * math.sqrt(T))
        vega = spot * math.exp(-q * T) * math.sqrt(T) * norm.pdf(d1) / 100
        
        return GreeksResult(
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(theta, 4),
            vega=round(vega, 4),
            rho=round(rho, 4),
            iv=round(iv * 100, 2),  # Convert back to percentage
            intrinsic_value=round(intrinsic, 2),
            time_value=round(time_val, 2)
        )
    
    def _calculate_iv_newton(
        self, spot, strike, T, r, q, market_price, opt_type, max_iter=100, tol=1e-5
    ) -> float:
        """Calculate IV using Newton-Raphson method"""
        from scipy.stats import norm
        
        sigma = 0.3  # Initial guess
        
        for _ in range(max_iter):
            d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if opt_type == "CE":
                price = spot * math.exp(-q * T) * norm.cdf(d1) - strike * math.exp(-r * T) * norm.cdf(d2)
            else:
                price = strike * math.exp(-r * T) * norm.cdf(-d2) - spot * math.exp(-q * T) * norm.cdf(-d1)
            
            vega = spot * math.exp(-q * T) * math.sqrt(T) * norm.pdf(d1)
            
            if vega < 1e-10:
                break
            
            diff = market_price - price
            if abs(diff) < tol:
                break
            
            sigma = sigma + diff / vega
            sigma = max(0.01, min(sigma, 5.0))  # Bound sigma
        
        return sigma
    
    # ========================================================================
    # BATCH CALCULATIONS
    # ========================================================================
    
    def calculate_chain_greeks(
        self,
        chain_df: pd.DataFrame,
        spot: float,
        days_to_expiry: int,
        strike_col: str = "StrikeRate",
        price_col: str = "LastRate",
        type_col: str = "CPType"
    ) -> pd.DataFrame:
        """
        Calculate Greeks for entire option chain
        
        Args:
            chain_df: DataFrame with option chain data
            spot: Current spot price
            days_to_expiry: Days to expiry
            strike_col: Column name for strike price
            price_col: Column name for option price
            type_col: Column name for option type
            
        Returns:
            DataFrame with Greeks columns added
        """
        if chain_df.empty:
            return chain_df
        
        greeks_list = []
        
        for _, row in chain_df.iterrows():
            greeks = self.calculate_greeks(
                spot=spot,
                strike=row[strike_col],
                days_to_expiry=days_to_expiry,
                option_price=row[price_col],
                option_type=row[type_col]
            )
            greeks_list.append(greeks.to_dict())
        
        greeks_df = pd.DataFrame(greeks_list)
        result = pd.concat([chain_df.reset_index(drop=True), greeks_df], axis=1)
        
        return result
    
    # ========================================================================
    # VOLATILITY ANALYSIS
    # ========================================================================
    
    def calculate_historical_volatility(
        self,
        prices: List[float],
        period: int = 20
    ) -> float:
        """
        Calculate historical volatility from price series
        
        Args:
            prices: List of closing prices
            period: Lookback period (default 20 days)
            
        Returns:
            Annualized historical volatility as percentage
        """
        if len(prices) < period + 1:
            return 0.0
        
        prices = np.array(prices[-period-1:])
        returns = np.log(prices[1:] / prices[:-1])
        
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)  # Annualize
        
        return round(annual_vol * 100, 2)
    
    def calculate_iv_percentile(
        self,
        current_iv: float,
        historical_ivs: List[float]
    ) -> float:
        """
        Calculate IV percentile (where current IV ranks historically)
        
        Args:
            current_iv: Current implied volatility
            historical_ivs: List of historical IV values
            
        Returns:
            Percentile (0-100)
        """
        if not historical_ivs:
            return 50.0
        
        below = sum(1 for iv in historical_ivs if iv < current_iv)
        percentile = (below / len(historical_ivs)) * 100
        
        return round(percentile, 1)
    
    def calculate_vol_skew(
        self,
        chain_df: pd.DataFrame,
        spot: float,
        option_type: str = "PE"
    ) -> Dict[str, float]:
        """
        Calculate volatility skew
        
        Args:
            chain_df: Option chain with IV calculated
            spot: Current spot price
            option_type: "PE" for put skew, "CE" for call skew
            
        Returns:
            Dictionary with skew metrics
        """
        if chain_df.empty or 'iv' not in chain_df.columns:
            return {"skew": 0, "atm_iv": 0, "otm_iv": 0}
        
        options = chain_df[chain_df['CPType'] == option_type].copy()
        
        if options.empty:
            return {"skew": 0, "atm_iv": 0, "otm_iv": 0}
        
        # Find ATM strike
        options['distance'] = abs(options['StrikeRate'] - spot)
        atm_row = options.loc[options['distance'].idxmin()]
        atm_iv = atm_row['iv']
        
        # Find OTM strike (5% away)
        if option_type == "PE":
            otm_strike = spot * 0.95
        else:
            otm_strike = spot * 1.05
        
        options['otm_distance'] = abs(options['StrikeRate'] - otm_strike)
        otm_row = options.loc[options['otm_distance'].idxmin()]
        otm_iv = otm_row['iv']
        
        skew = otm_iv - atm_iv
        
        return {
            "skew": round(skew, 2),
            "atm_iv": round(atm_iv, 2),
            "otm_iv": round(otm_iv, 2),
            "atm_strike": atm_row['StrikeRate'],
            "otm_strike": otm_row['StrikeRate']
        }
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_delta_strike(
        self,
        chain_df: pd.DataFrame,
        target_delta: float,
        option_type: str = "CE"
    ) -> Optional[Dict]:
        """
        Find strike closest to target delta
        
        Args:
            chain_df: Option chain with delta calculated
            target_delta: Target delta (e.g., 0.30 for 30-delta)
            option_type: "CE" or "PE"
            
        Returns:
            Dictionary with strike details or None
        """
        if chain_df.empty or 'delta' not in chain_df.columns:
            return None
        
        options = chain_df[chain_df['CPType'] == option_type].copy()
        
        if options.empty:
            return None
        
        # For puts, delta is negative, so we compare absolute values
        if option_type == "PE":
            options['delta_diff'] = abs(abs(options['delta']) - abs(target_delta))
        else:
            options['delta_diff'] = abs(options['delta'] - target_delta)
        
        closest = options.loc[options['delta_diff'].idxmin()]
        
        return {
            "strike": closest['StrikeRate'],
            "delta": closest['delta'],
            "price": closest['LastRate'],
            "iv": closest.get('iv', 0)
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Greeks Calculator Test")
    print("=" * 50)
    
    calc = GreeksCalculator()
    
    # Test 1: Basic Greeks calculation
    print("\n1. Basic Greeks Calculation:")
    greeks = calc.calculate_greeks(
        spot=24000,
        strike=24000,
        days_to_expiry=7,
        option_price=250,
        option_type="CE"
    )
    print(f"   Delta: {greeks.delta}")
    print(f"   Gamma: {greeks.gamma}")
    print(f"   Theta: {greeks.theta}")
    print(f"   Vega: {greeks.vega}")
    print(f"   IV: {greeks.iv}%")
    print(f"   Intrinsic: {greeks.intrinsic_value}")
    print(f"   Time Value: {greeks.time_value}")
    
    # Test 2: Put option
    print("\n2. Put Option Greeks:")
    put_greeks = calc.calculate_greeks(
        spot=24000,
        strike=23500,
        days_to_expiry=7,
        option_price=120,
        option_type="PE"
    )
    print(f"   Delta: {put_greeks.delta}")
    print(f"   IV: {put_greeks.iv}%")
    
    # Test 3: Historical volatility
    print("\n3. Historical Volatility:")
    import random
    prices = [24000 + random.uniform(-500, 500) for _ in range(30)]
    hv = calc.calculate_historical_volatility(prices)
    print(f"   HV (20-day): {hv}%")
    
    # Test 4: IV Percentile
    print("\n4. IV Percentile:")
    historical_ivs = [random.uniform(10, 30) for _ in range(252)]
    current_iv = 22.5
    percentile = calc.calculate_iv_percentile(current_iv, historical_ivs)
    print(f"   Current IV: {current_iv}%")
    print(f"   IV Percentile: {percentile}")
    
    print("\n✅ Greeks Calculator test complete!")
