"""
RedTeam Semantic Caching System

Provides intelligent caching based on semantic similarity to reduce token usage
and improve performance through cached responses for similar requests.
"""

from .semantic_cache import (
    SemanticCacheManager,
    SemanticCacheConfig,
    CacheEntry,
    get_cache_manager,
    cache_response,
    get_cached_response,
    get_cache_stats
)

__all__ = [
    "SemanticCacheManager",
    "SemanticCacheConfig", 
    "CacheEntry",
    "get_cache_manager",
    "cache_response",
    "get_cached_response",
    "get_cache_stats"
]