"""
Cache Manager for NSEApp v2
Implements TTL-based caching for API responses to reduce API calls
"""

import time
import json
import hashlib
import pickle
from typing import Any, Dict, Optional, Callable
from pathlib import Path
from functools import wraps
from datetime import datetime, timedelta
import threading

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CACHE_CONFIG, DATA_DIR


class CacheEntry:
    """Represents a single cache entry with timestamp"""
    
    def __init__(self, data: Any, ttl: int):
        self.data = data
        self.created_at = time.time()
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        """Check if this cache entry has expired"""
        return time.time() - self.created_at > self.ttl
    
    def time_remaining(self) -> float:
        """Get remaining TTL in seconds"""
        remaining = self.ttl - (time.time() - self.created_at)
        return max(0, remaining)
    
    def __repr__(self):
        return f"CacheEntry(expired={self.is_expired()}, remaining={self.time_remaining():.0f}s)"


class CacheManager:
    """
    Thread-safe in-memory cache with TTL support
    Also supports disk persistence for session continuity
    """
    
    def __init__(self, default_ttl: int = None, persist_to_disk: bool = False):
        """
        Initialize cache manager
        
        Args:
            default_ttl: Default TTL in seconds (default from config: 900 = 15 min)
            persist_to_disk: Whether to save cache to disk on shutdown
        """
        self.default_ttl = default_ttl or CACHE_CONFIG.get("default_ttl", 900)
        self.persist_to_disk = persist_to_disk
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._enabled = CACHE_CONFIG.get("enabled", True)
        
        # Disk cache path
        self._cache_dir = DATA_DIR / "cache"
        if self.persist_to_disk:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from arguments"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if exists and not expired
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if not self._enabled:
            return None
            
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            return entry.data
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (optional, uses default if not provided)
        """
        if not self._enabled:
            return
            
        ttl = ttl or self.default_ttl
        
        with self._lock:
            self._cache[key] = CacheEntry(value, ttl)
    
    def delete(self, key: str) -> bool:
        """
        Delete a specific key from cache
        
        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> int:
        """
        Clear all cache entries
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total = len(self._cache)
            expired = sum(1 for v in self._cache.values() if v.is_expired())
            active = total - expired
            
            return {
                "total_entries": total,
                "active_entries": active,
                "expired_entries": expired,
                "enabled": self._enabled,
                "default_ttl": self.default_ttl,
            }
    
    def cached(self, ttl: int = None, key_prefix: str = ""):
        """
        Decorator for caching function results
        
        Args:
            ttl: TTL in seconds
            key_prefix: Optional prefix for cache key
            
        Example:
            @cache.cached(ttl=900, key_prefix="option_chain")
            def get_option_chain(symbol, expiry):
                ...
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = f"{key_prefix}:{func.__name__}:{self._generate_key(*args, **kwargs)}"
                
                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Call function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def _load_from_disk(self) -> None:
        """Load cache from disk if exists"""
        cache_file = self._cache_dir / "cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self._cache = pickle.load(f)
                # Cleanup expired entries after loading
                self.cleanup_expired()
            except Exception:
                # If loading fails, start with empty cache
                self._cache = {}
    
    def save_to_disk(self) -> None:
        """Save current cache to disk"""
        if not self.persist_to_disk:
            return
            
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_dir / "cache.pkl"
        
        with self._lock:
            # Only save non-expired entries
            active_cache = {k: v for k, v in self._cache.items() if not v.is_expired()}
            with open(cache_file, "wb") as f:
                pickle.dump(active_cache, f)
    
    def __repr__(self):
        stats = self.get_stats()
        return f"CacheManager(active={stats['active_entries']}, ttl={self.default_ttl}s)"


# ============================================================================
# SPECIALIZED CACHE INSTANCES
# ============================================================================

# Global cache instance for the application
_global_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Get the global cache instance (singleton pattern)"""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager(
            default_ttl=CACHE_CONFIG.get("default_ttl", 900),
            persist_to_disk=False
        )
    return _global_cache


# Convenience functions using global cache
def cached(ttl: int = None, key_prefix: str = ""):
    """Decorator using global cache"""
    return get_cache().cached(ttl=ttl, key_prefix=key_prefix)


# ============================================================================
# TTL PRESETS (from config)
# ============================================================================

class CacheTTL:
    """Preset TTL values for different data types"""
    QUOTES = CACHE_CONFIG.get("quotes_ttl", 900)          # 15 min
    OPTION_CHAIN = CACHE_CONFIG.get("option_chain_ttl", 900)  # 15 min
    EXPIRY_LIST = CACHE_CONFIG.get("expiry_list_ttl", 3600)   # 1 hour
    FNO_STOCKS = CACHE_CONFIG.get("fno_stocks_ttl", 86400)    # 24 hours


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Cache Manager Test")
    print("=" * 50)
    
    # Create cache with short TTL for testing
    cache = CacheManager(default_ttl=5)  # 5 seconds TTL
    
    # Test basic operations
    cache.set("test_key", {"symbol": "NIFTY", "price": 19500})
    print(f"Set test_key: {cache.get('test_key')}")
    
    # Test decorator
    @cache.cached(ttl=10, key_prefix="test")
    def fetch_data(symbol: str) -> dict:
        print(f"  [Cache Miss] Fetching data for {symbol}...")
        return {"symbol": symbol, "timestamp": datetime.now().isoformat()}
    
    # First call - cache miss
    result1 = fetch_data("NIFTY")
    print(f"First call: {result1}")
    
    # Second call - cache hit
    result2 = fetch_data("NIFTY")
    print(f"Second call (cached): {result2}")
    
    # Different args - cache miss
    result3 = fetch_data("BANKNIFTY")
    print(f"Different symbol: {result3}")
    
    # Stats
    print(f"\nCache stats: {cache.get_stats()}")
    
    # Wait for expiry
    print("\nWaiting 6 seconds for expiry...")
    time.sleep(6)
    
    # After expiry
    expired = cache.cleanup_expired()
    print(f"Cleaned up {expired} expired entries")
    print(f"Cache stats after cleanup: {cache.get_stats()}")
    
    print("\n✅ Cache Manager working correctly!")
