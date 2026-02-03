"""
NSE F&O Stocks Universe
Complete list of ALL stocks trading in NSE F&O segment (~180+ stocks)
"""

from typing import List, Dict, Set

# ============================================================================
# INDICES (F&O Enabled)
# ============================================================================

NSE_INDICES = [
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
]

# ============================================================================
# COMPLETE NSE F&O STOCKS LIST (As of Jan 2026)
# ============================================================================

NSE_FNO_STOCKS = [
    # A
    "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", 
    "ADANIENT", "ADANIPORTS", "ADANIPOWER", "ALKEM", "AMBUJACEM", 
    "ANGELONE", "APLAPOLLO", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY",
    "ASIANPAINT", "ASTRAL", "ATUL", "AUBANK", "AUROPHARMA", "AXISBANK",
    
    # B
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BALKRISIND", "BALRAMCHIN",
    "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BATAINDIA", "BDL", "BEL",
    "BERGEPAINT", "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", 
    "BOSCHLTD", "BPCL", "BRITANNIA", "BSE", "BSOFT",
    
    # C
    "CAMS", "CANBK", "CANFINHOME", "CDSL", "CESC", "CGPOWER",
    "CHAMBLFERT", "CHOLAFIN", "CIPLA", "CLEAN", "COALINDIA", "COFORGE",
    "COLPAL", "CONCOR", "COROMANDEL", "CROMPTON", "CUB", "CUMMINSIND",
    
    # D
    "DABUR", "DALBHARAT", "DEEPAKNTR", "DELHIVERY", "DELTACORP",
    "DEVYANI", "DIVISLAB", "DIXON", "DLF", "DRREDDY",
    
    # E
    "EICHERMOT", "ESCORTS", "ETERNAL", "EXIDEIND",
    
    # F
    "FEDERALBNK", "FORTIS", "FSL",
    
    # G
    "GAIL", "GLENMARK", "GMRAIRPORT", "GNFC", "GODREJCP", "GODREJPROP",
    "GRANULES", "GRAPHITE", "GRASIM", "GSPL", "GUJGASLTD",
    
    # H
    "HAL", "HAVELLS", "HCLTECH", "HDFC", "HDFCAMC", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDCOPPER", "HINDPETRO",
    "HINDUNILVR", "HONAUT", "HUDCO",
    
    # I
    "IBULHSGFIN", "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDEA",
    "IDFC", "IDFCFIRSTB", "IEX", "IGL", "IIFL", "INDHOTEL",
    "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY",
    "IOC", "IPCALAB", "IRCTC", "IRFC", "ITC", "ITDC",
    
    # J
    "JINDALSTEL", "JIOFIN", "JKCEMENT", "JSWENERGY", "JSWSTEEL", "JUBLFOOD",
    
    # K
    "KALYANKJIL", "KEI", "KOTAKBANK", "KPITTECH",
    
    # L
    "L&TFH", "LALPATHLAB", "LAURUSLABS", "LICHSGFIN", "LT", "LTIM",
    "LTTS", "LUPIN",
    
    # M
    "M&M", "M&MFIN", "MAHABANK", "MANAPPURAM", "MARICO", "MARUTI",
    "MAXHEALTH", "MCX", "METROPOLIS", "MFSL", "MGL", "MOTHERSON",
    "MPHASIS", "MRF", "MUTHOOTFIN",
    
    # N
    "NATIONALUM", "NAUKRI", "NAVINFLUOR", "NCC", "NESTLEIND", "NHPC",
    "NMDC", "NTPC", "NYKAA",
    
    # O
    "OBEROIRLTY", "OFSS", "OIL", "ONGC",
    
    # P
    "PAGEIND", "PATANJALI", "PAYTM", "PEL", "PERSISTENT", "PETRONET",
    "PFC", "PIDILITIND", "PIIND", "PNB", "POLYCAB", "POONAWALLA",
    "POWERGRID", "PRESTIGE", "PVRINOX",
    
    # R
    "RAIN", "RAMCOCEM", "RBLBANK", "RECLTD", "RELIANCE", "RVNL",
    
    # S
    "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM", "SHRIRAMFIN",
    "SIEMENS", "SJVN", "SOBHA", "SOLARINDS", "SONACOMS", "SRF",
    "STAR", "SUNPHARMA", "SUNDARMFIN", "SUNDRMFAST", "SUPREMEIND",
    "SYNGENE", "SYRMA",
    
    # T
    "TATACHEM", "TATACOMM", "TATACONSUM", "TATAELXSI", "TATAMOTORS",
    "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "THERMAX", "TITAN",
    "TORNTPHARM", "TORNTPOWER", "TRENT", "TRIDENT", "TRIVENI", "TTML",
    "TVS",
    
    # U
    "UBL", "ULTRACEMCO", "UNIONBANK", "UPL",
    
    # V
    "VBL", "VEDL", "VOLTAS",
    
    # W
    "WIPRO",
    
    # Y
    "YESBANK",
    
    # Z
    "ZEEL", "ZOMATO", "ZYDUSLIFE",
]

# ============================================================================
# SECTOR MAPPING (Priority sectors first)
# ============================================================================

SECTOR_MAP: Dict[str, List[str]] = {
    # PRIORITY SECTORS
    "Financials": [
        "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "BANKBARODA",
        "INDUSINDBK", "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "PNB", "CANBK",
        "AUBANK", "RBLBANK", "YESBANK", "IDFC", "CUB", "MAHABANK", "INDIANB",
        "UNIONBANK", "BANKINDIA",
        # NBFCs
        "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "SHRIRAMFIN", "M&MFIN", "L&TFH",
        "LICHSGFIN", "CANFINHOME", "POONAWALLA", "MUTHOOTFIN", "MANAPPURAM",
        "IIFL", "IBULHSGFIN",
        # Insurance
        "HDFCLIFE", "SBILIFE", "ICICIPRULI", "ICICIGI", "STAR",
        # AMC & Financial Services
        "HDFCAMC", "CAMS", "CDSL", "BSE", "MCX", "IEX", "ANGELONE", "SBICARD",
        "JIOFIN", "MFSL",
    ],
    
    "Infrastructure": [
        "LT", "ADANIPORTS", "ADANIENT", "DLF", "GODREJPROP", "OBEROIRLTY",
        "PRESTIGE", "SOBHA", "CONCOR", "GMRAIRPORT", "IRFC", "HUDCO",
        "RVNL", "NCC", "APLAPOLLO", "KEI", "POLYCAB", "CGPOWER",
        "INDUSTOWER", "DELHIVERY", "IRCTC",
    ],
    
    "Renewable Energy": [
        "ADANIPOWER", "TATAPOWER", "NTPC", "POWERGRID", "JSWENERGY",
        "SJVN", "NHPC", "RECLTD", "PFC", "TORNTPOWER", "CESC",
        "CLEAN",
    ],
    
    "Automotive": [
        "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT",
        "ASHOKLEY", "TVS", "ESCORTS", "MOTHERSON", "BHARATFORG", "BALKRISIND",
        "APOLLOTYRE", "MRF", "EXIDEIND", "BOSCHLTD", "SONACOMS", "SUNDRMFAST",
    ],
    
    "Healthcare": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA", "LUPIN",
        "BIOCON", "TORNTPHARM", "ALKEM", "IPCALAB", "LAURUSLABS", "GLENMARK",
        "GRANULES", "ZYDUSLIFE", "ABBOTINDIA", "LALPATHLAB", "METROPOLIS",
        "APOLLOHOSP", "FORTIS", "MAXHEALTH", "SYNGENE",
    ],
    
    # OTHER SECTORS
    "IT": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "LTTS",
        "MPHASIS", "COFORGE", "PERSISTENT", "KPITTECH", "TATAELXSI",
        "OFSS", "NAUKRI", "BSOFT", "FSL",
    ],
    
    "FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO",
        "COLPAL", "GODREJCP", "TATACONSUM", "VBL", "UBL", "PATANJALI",
        "DEVYANI", "JUBLFOOD", "ZOMATO", "NYKAA", "TRENT", "TITAN",
        "PAGEIND", "BATAINDIA",
    ],
    
    "Metals & Mining": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA", "NMDC",
        "SAIL", "JINDALSTEL", "NATIONALUM", "HINDCOPPER", "GRAPHITE",
        "RAIN",
    ],
    
    "Oil & Gas": [
        "RELIANCE", "ONGC", "BPCL", "IOC", "HINDPETRO", "GAIL", "PETRONET",
        "OIL", "GSPL", "GUJGASLTD", "IGL", "MGL",
    ],
    
    "Cement & Building Materials": [
        "ULTRACEMCO", "SHREECEM", "AMBUJACEM", "ACC", "DALBHARAT", "RAMCOCEM",
        "JKCEMENT", "PIDILITIND", "ASTRAL", "SUPREMEIND",
    ],
    
    "Chemicals": [
        "PIDILITIND", "SRF", "AARTIIND", "DEEPAKNTR", "NAVINFLUOR", "ATUL",
        "COROMANDEL", "CHAMBLFERT", "GNFC", "PIIND", "CLEAN", "TATACHEM",
    ],
    
    "Telecom": [
        "BHARTIARTL", "IDEA", "TATACOMM", "TTML",
    ],
    
    "Consumer Durables": [
        "HAVELLS", "VOLTAS", "CROMPTON", "DIXON", "TITAN", "HONAUT",
        "BLUESTARCO", "KALYANKJIL", "PVRINOX",
    ],
    
    "Defence": [
        "HAL", "BEL", "BDL", "COCHINSHIP",
    ],
    
    "Hospitality & Travel": [
        "INDHOTEL", "INDIGO", "IRCTC",
    ],
    
    "Media & Entertainment": [
        "ZEEL", "PVR",
    ],
    
    "Miscellaneous": [
        "ETERNAL", "PAYTM", "DELTACORP", "TRIDENT", "TRIVENI", "THERMAX",
        "CUMMINSIND", "SIEMENS", "ABB", "SYRMA",
    ],
}

# ============================================================================
# PRIORITY SECTORS (As specified by user)
# ============================================================================

PRIORITY_SECTORS = [
    "Financials",
    "Infrastructure",
    "Renewable Energy",
    "Automotive",
    "Healthcare",
]

# ============================================================================
# NIFTY INDICES CONSTITUENTS
# ============================================================================

NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK",
    "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA",
    "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS", "TECHM",
    "TITAN", "ULTRACEMCO", "UPL", "WIPRO", "ETERNAL",
]

NIFTY_BANK = [
    "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK",
    "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB",
    "BANKBARODA", "AUBANK",
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_fno_symbols() -> List[str]:
    """Get all FNO symbols including indices"""
    return NSE_INDICES + NSE_FNO_STOCKS


def get_stocks_by_sector(sector: str) -> List[str]:
    """Get stocks belonging to a specific sector"""
    return SECTOR_MAP.get(sector, [])


def get_sector_for_stock(symbol: str) -> str:
    """Get sector for a given stock symbol"""
    for sector, stocks in SECTOR_MAP.items():
        if symbol in stocks:
            return sector
    return "Unknown"


def get_priority_sector_stocks() -> List[str]:
    """Get all stocks from priority sectors"""
    stocks = []
    for sector in PRIORITY_SECTORS:
        stocks.extend(SECTOR_MAP.get(sector, []))
    return list(set(stocks))  # Remove duplicates


def get_all_sectors() -> List[str]:
    """Get list of all sectors"""
    return list(SECTOR_MAP.keys())


def is_index(symbol: str) -> bool:
    """Check if symbol is an index"""
    return symbol in NSE_INDICES


def is_fno_stock(symbol: str) -> bool:
    """Check if symbol is in FNO segment"""
    return symbol in NSE_FNO_STOCKS or symbol in NSE_INDICES


def get_stock_count() -> Dict[str, int]:
    """Get count of stocks by category"""
    return {
        "total_fno": len(NSE_FNO_STOCKS),
        "indices": len(NSE_INDICES),
        "total_symbols": len(get_all_fno_symbols()),
        "sectors": len(SECTOR_MAP),
        "priority_sector_stocks": len(get_priority_sector_stocks()),
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("NSE F&O Stock Universe")
    print("=" * 50)
    
    counts = get_stock_count()
    print(f"\nTotal FNO Stocks: {counts['total_fno']}")
    print(f"Indices: {counts['indices']}")
    print(f"Total Symbols: {counts['total_symbols']}")
    print(f"Sectors: {counts['sectors']}")
    print(f"Priority Sector Stocks: {counts['priority_sector_stocks']}")
    
    print("\n\nPriority Sectors:")
    for sector in PRIORITY_SECTORS:
        stocks = get_stocks_by_sector(sector)
        print(f"  {sector}: {len(stocks)} stocks")
    
    print("\n\nAll Sectors:")
    for sector in get_all_sectors():
        stocks = get_stocks_by_sector(sector)
        print(f"  {sector}: {len(stocks)} stocks")
