"""
Buildup Classifier for NSEApp v2
Classifies price/OI changes into Long, Long Unwinding, Short, Short Covering
"""

from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BUILDUP_COLORS, BUILDUP_TEXT_COLORS


class BuildupType(Enum):
    """Enumeration of buildup types"""
    LONG = "Long"                    # Price up + OI up = Fresh buying (bullish)
    LONG_UNWINDING = "LU"            # Price down + OI down = Longs exiting (bearish)
    SHORT = "Short"                  # Price down + OI up = Fresh selling (bearish)
    SHORT_COVERING = "SC"            # Price up + OI down = Shorts exiting (bullish)
    NEUTRAL = "Neutral"              # No significant change


# Short aliases for display
BUILDUP_SHORT_NAMES = {
    BuildupType.LONG: "Long",
    BuildupType.LONG_UNWINDING: "LU",
    BuildupType.SHORT: "Short",
    BuildupType.SHORT_COVERING: "SC",
    BuildupType.NEUTRAL: "-"
}

# Detailed descriptions
BUILDUP_DESCRIPTIONS = {
    BuildupType.LONG: "Fresh buying - Traders opening new long positions (Bullish)",
    BuildupType.LONG_UNWINDING: "Longs exiting - Profit booking or stop loss (Bearish)",
    BuildupType.SHORT: "Fresh selling - Traders opening new short positions (Bearish)",
    BuildupType.SHORT_COVERING: "Shorts exiting - Profit booking or stop loss (Bullish)",
    BuildupType.NEUTRAL: "No significant buildup activity"
}

# Market sentiment
BUILDUP_SENTIMENT = {
    BuildupType.LONG: "Bullish",
    BuildupType.LONG_UNWINDING: "Bearish",
    BuildupType.SHORT: "Bearish",
    BuildupType.SHORT_COVERING: "Bullish",
    BuildupType.NEUTRAL: "Neutral"
}


@dataclass
class BuildupResult:
    """Result of buildup classification"""
    symbol: str
    buildup_type: BuildupType
    price_change: float
    price_change_pct: float
    oi_change: int
    oi_change_pct: float
    sentiment: str
    color: str
    text_color: str
    description: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "buildup": self.buildup_type.value,
            "short_name": BUILDUP_SHORT_NAMES[self.buildup_type],
            "price_change": self.price_change,
            "price_change_pct": self.price_change_pct,
            "oi_change": self.oi_change,
            "oi_change_pct": self.oi_change_pct,
            "sentiment": self.sentiment,
            "color": self.color,
            "text_color": self.text_color,
            "description": self.description
        }


def classify_buildup(
    price_change: float, 
    oi_change: float,
    price_change_threshold: float = 0.0,
    oi_change_threshold: float = 0.0
) -> BuildupType:
    """
    Classify buildup based on price and OI changes
    
    Logic:
    - Price ↑ + OI ↑ = Long (Fresh buying, bullish)
    - Price ↓ + OI ↓ = Long Unwinding (Longs exiting, bearish)
    - Price ↓ + OI ↑ = Short (Fresh selling, bearish)
    - Price ↑ + OI ↓ = Short Covering (Shorts exiting, bullish)
    
    Args:
        price_change: Price change amount or percentage
        oi_change: OI change amount or percentage
        price_change_threshold: Minimum threshold for significant price change
        oi_change_threshold: Minimum threshold for significant OI change
        
    Returns:
        BuildupType enum value
    """
    # Check if changes are significant
    price_significant = abs(price_change) > price_change_threshold
    oi_significant = abs(oi_change) > oi_change_threshold
    
    if not (price_significant and oi_significant):
        return BuildupType.NEUTRAL
    
    price_up = price_change > 0
    oi_up = oi_change > 0
    
    if price_up and oi_up:
        return BuildupType.LONG
    elif not price_up and not oi_up:
        return BuildupType.LONG_UNWINDING
    elif not price_up and oi_up:
        return BuildupType.SHORT
    elif price_up and not oi_up:
        return BuildupType.SHORT_COVERING
    else:
        return BuildupType.NEUTRAL


def classify_with_details(
    symbol: str,
    current_price: float,
    previous_price: float,
    current_oi: int,
    previous_oi: int,
    price_threshold_pct: float = 0.1,
    oi_threshold_pct: float = 0.5
) -> BuildupResult:
    """
    Classify buildup with full details
    
    Args:
        symbol: Stock/Index symbol
        current_price: Current LTP
        previous_price: Previous day close
        current_oi: Current Open Interest
        previous_oi: Previous day OI
        price_threshold_pct: Minimum price change % for significance
        oi_threshold_pct: Minimum OI change % for significance
        
    Returns:
        BuildupResult with all details
    """
    # Calculate changes
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price * 100) if previous_price > 0 else 0
    
    oi_change = current_oi - previous_oi
    oi_change_pct = (oi_change / previous_oi * 100) if previous_oi > 0 else 0
    
    # Classify
    buildup_type = classify_buildup(
        price_change_pct, 
        oi_change_pct,
        price_threshold_pct,
        oi_threshold_pct
    )
    
    return BuildupResult(
        symbol=symbol,
        buildup_type=buildup_type,
        price_change=round(price_change, 2),
        price_change_pct=round(price_change_pct, 2),
        oi_change=oi_change,
        oi_change_pct=round(oi_change_pct, 2),
        sentiment=BUILDUP_SENTIMENT[buildup_type],
        color=BUILDUP_COLORS.get(buildup_type.value, "#9E9E9E"),
        text_color=BUILDUP_TEXT_COLORS.get(buildup_type.value, "#FFFFFF"),
        description=BUILDUP_DESCRIPTIONS[buildup_type]
    )


def classify_dataframe(
    df: pd.DataFrame,
    price_col: str = "ltp",
    prev_price_col: str = "close",
    oi_col: str = "oi",
    prev_oi_col: str = "prev_oi",
    symbol_col: str = "symbol"
) -> pd.DataFrame:
    """
    Classify buildup for an entire DataFrame
    
    Args:
        df: DataFrame with price and OI data
        price_col: Column name for current price
        prev_price_col: Column name for previous close
        oi_col: Column name for current OI
        prev_oi_col: Column name for previous OI
        symbol_col: Column name for symbol
        
    Returns:
        DataFrame with buildup classification columns added
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Calculate changes
    df['price_change'] = df[price_col] - df[prev_price_col]
    df['price_change_pct'] = (df['price_change'] / df[prev_price_col] * 100).round(2)
    
    df['oi_change'] = df[oi_col] - df[prev_oi_col]
    df['oi_change_pct'] = (df['oi_change'] / df[prev_oi_col] * 100).round(2)
    
    # Classify each row
    def classify_row(row):
        buildup = classify_buildup(row['price_change_pct'], row['oi_change_pct'])
        return pd.Series({
            'buildup': buildup.value,
            'buildup_short': BUILDUP_SHORT_NAMES[buildup],
            'sentiment': BUILDUP_SENTIMENT[buildup],
            'buildup_color': BUILDUP_COLORS.get(buildup.value, "#9E9E9E"),
            'text_color': BUILDUP_TEXT_COLORS.get(buildup.value, "#FFFFFF")
        })
    
    buildup_df = df.apply(classify_row, axis=1)
    result = pd.concat([df, buildup_df], axis=1)
    
    return result


def get_buildup_summary(classified_df: pd.DataFrame) -> Dict[str, int]:
    """
    Get summary counts of each buildup type
    
    Args:
        classified_df: DataFrame with 'buildup' column
        
    Returns:
        Dictionary with counts for each buildup type
    """
    if classified_df.empty or 'buildup' not in classified_df.columns:
        return {bt.value: 0 for bt in BuildupType}
    
    counts = classified_df['buildup'].value_counts().to_dict()
    
    # Ensure all types are present
    result = {bt.value: 0 for bt in BuildupType}
    result.update(counts)
    
    return result


def filter_by_buildup(
    classified_df: pd.DataFrame, 
    buildup_types: List[BuildupType]
) -> pd.DataFrame:
    """
    Filter DataFrame by specific buildup types
    
    Args:
        classified_df: DataFrame with 'buildup' column
        buildup_types: List of BuildupType to filter
        
    Returns:
        Filtered DataFrame
    """
    if classified_df.empty or 'buildup' not in classified_df.columns:
        return classified_df
    
    type_values = [bt.value for bt in buildup_types]
    return classified_df[classified_df['buildup'].isin(type_values)]


def get_strongest_buildup(
    classified_df: pd.DataFrame,
    buildup_type: BuildupType,
    top_n: int = 10,
    sort_by: str = "oi_change_pct"
) -> pd.DataFrame:
    """
    Get stocks with strongest buildup of a specific type
    
    Args:
        classified_df: DataFrame with buildup classification
        buildup_type: Type of buildup to filter
        top_n: Number of top results
        sort_by: Column to sort by (oi_change_pct or price_change_pct)
        
    Returns:
        Top N stocks with specified buildup
    """
    filtered = filter_by_buildup(classified_df, [buildup_type])
    
    if filtered.empty:
        return filtered
    
    # Sort based on buildup type
    if buildup_type in [BuildupType.LONG, BuildupType.SHORT]:
        # For Long and Short, higher OI change is stronger
        return filtered.nlargest(top_n, sort_by)
    else:
        # For Unwinding and Covering, larger negative OI change is stronger
        return filtered.nsmallest(top_n, sort_by)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Buildup Classifier Test")
    print("=" * 50)
    
    # Test classification logic
    test_cases = [
        (5.0, 10.0, "Long"),
        (-3.0, -5.0, "LU"),
        (-2.0, 8.0, "Short"),
        (4.0, -6.0, "SC"),
        (0.05, 0.1, "Neutral"),
    ]
    
    print("\nClassification Logic Test:")
    for price_chg, oi_chg, expected in test_cases:
        result = classify_buildup(price_chg, oi_chg)
        status = "✅" if BUILDUP_SHORT_NAMES[result] == expected else "❌"
        print(f"  Price {price_chg:+.1f}%, OI {oi_chg:+.1f}% => {BUILDUP_SHORT_NAMES[result]} {status}")
    
    # Test with details
    print("\nDetailed Classification Test:")
    result = classify_with_details(
        symbol="RELIANCE",
        current_price=2850.50,
        previous_price=2800.00,
        current_oi=5000000,
        previous_oi=4500000
    )
    print(f"  {result.symbol}: {result.buildup_type.value}")
    print(f"  Price: {result.price_change_pct:+.2f}%")
    print(f"  OI: {result.oi_change_pct:+.2f}%")
    print(f"  Sentiment: {result.sentiment}")
    print(f"  Color: {result.color}")
    
    # Test DataFrame classification
    print("\nDataFrame Classification Test:")
    test_df = pd.DataFrame({
        'symbol': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK'],
        'ltp': [2850, 3500, 1650, 1580],
        'close': [2800, 3550, 1620, 1600],
        'oi': [5000000, 3000000, 4000000, 6000000],
        'prev_oi': [4500000, 3200000, 3800000, 5800000]
    })
    
    classified = classify_dataframe(test_df)
    print(classified[['symbol', 'price_change_pct', 'oi_change_pct', 'buildup', 'sentiment']])
    
    print("\nBuildup Summary:")
    summary = get_buildup_summary(classified)
    for buildup_type, count in summary.items():
        print(f"  {buildup_type}: {count}")
    
    print("\n✅ Buildup Classifier tests complete!")
