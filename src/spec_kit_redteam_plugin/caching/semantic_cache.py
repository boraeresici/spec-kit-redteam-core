#!/usr/bin/env python3
"""
RedTeam Semantic Caching System

Provides intelligent caching based on semantic similarity to reduce token usage
and improve performance through cached responses for similar requests.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json
import pickle
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import threading
import logging


@dataclass
class CacheEntry:
    """Semantic cache entry with metadata."""
    
    cache_key: str
    request_hash: str
    semantic_embedding: np.ndarray
    request_data: Dict[str, Any]
    response_data: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    similarity_threshold: float = 0.85
    expiry_hours: int = 24
    context_size: int = 0
    token_savings: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.utcnow() > self.created_at + timedelta(hours=self.expiry_hours)
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
    
    def calculate_similarity(self, other_embedding: np.ndarray) -> float:
        """Calculate semantic similarity with another embedding."""
        return cosine_similarity(
            self.semantic_embedding.reshape(1, -1),
            other_embedding.reshape(1, -1)
        )[0][0]


class SemanticCacheConfig:
    """Configuration for semantic caching system."""
    
    def __init__(self):
        self.similarity_threshold = 0.85
        self.max_cache_size = 1000
        self.default_expiry_hours = 24
        self.embedding_model = "all-MiniLM-L6-v2"
        self.cache_db_path = "semantic_cache.db"
        self.enable_persistence = True
        self.cleanup_interval_hours = 6
        self.min_token_threshold = 50  # Minimum tokens to cache
        self.max_context_size = 8000   # Max context size to cache


class SemanticCacheManager:
    """Manages semantic similarity-based caching for AI requests."""
    
    def __init__(self, config: Optional[SemanticCacheConfig] = None):
        self.config = config or SemanticCacheConfig()
        self.cache: Dict[str, CacheEntry] = {}
        self.embedding_model = None
        self.db_connection = None
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "token_savings": 0,
            "entries_created": 0,
            "entries_expired": 0
        }
        
        self._initialize_model()
        self._initialize_database()
        self._load_cache_from_db()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            logging.info(f"Loaded embedding model: {self.config.embedding_model}")
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            raise
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent caching."""
        if not self.config.enable_persistence:
            return
            
        try:
            self.db_connection = sqlite3.connect(
                self.config.cache_db_path, 
                check_same_thread=False
            )
            
            # Create tables
            self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    request_hash TEXT,
                    semantic_embedding BLOB,
                    request_data TEXT,
                    response_data TEXT,
                    created_at TIMESTAMP,
                    last_accessed TIMESTAMP,
                    access_count INTEGER,
                    similarity_threshold REAL,
                    expiry_hours INTEGER,
                    context_size INTEGER,
                    token_savings INTEGER
                )
            """)
            
            self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS cache_stats (
                    stat_name TEXT PRIMARY KEY,
                    stat_value INTEGER
                )
            """)
            
            self.db_connection.commit()
            
        except Exception as e:
            logging.error(f"Failed to initialize database: {e}")
            self.config.enable_persistence = False
    
    def _load_cache_from_db(self):
        """Load existing cache entries from database."""
        if not self.config.enable_persistence or not self.db_connection:
            return
        
        try:
            cursor = self.db_connection.execute("""
                SELECT * FROM cache_entries 
                WHERE created_at > datetime('now', '-' || expiry_hours || ' hours')
            """)
            
            for row in cursor.fetchall():
                cache_key, request_hash, embedding_blob, request_data, response_data, \
                created_at, last_accessed, access_count, similarity_threshold, \
                expiry_hours, context_size, token_savings = row
                
                entry = CacheEntry(
                    cache_key=cache_key,
                    request_hash=request_hash,
                    semantic_embedding=pickle.loads(embedding_blob),
                    request_data=json.loads(request_data),
                    response_data=json.loads(response_data),
                    created_at=datetime.fromisoformat(created_at),
                    last_accessed=datetime.fromisoformat(last_accessed),
                    access_count=access_count,
                    similarity_threshold=similarity_threshold,
                    expiry_hours=expiry_hours,
                    context_size=context_size,
                    token_savings=token_savings
                )
                
                if not entry.is_expired():
                    self.cache[cache_key] = entry
            
            # Load statistics
            cursor = self.db_connection.execute("SELECT * FROM cache_stats")
            for stat_name, stat_value in cursor.fetchall():
                if stat_name in self.stats:
                    self.stats[stat_name] = stat_value
                    
            logging.info(f"Loaded {len(self.cache)} cache entries from database")
            
        except Exception as e:
            logging.error(f"Failed to load cache from database: {e}")
    
    def _save_entry_to_db(self, entry: CacheEntry):
        """Save cache entry to database."""
        if not self.config.enable_persistence or not self.db_connection:
            return
        
        try:
            self.db_connection.execute("""
                INSERT OR REPLACE INTO cache_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.cache_key,
                entry.request_hash,
                pickle.dumps(entry.semantic_embedding),
                json.dumps(entry.request_data),
                json.dumps(entry.response_data),
                entry.created_at.isoformat(),
                entry.last_accessed.isoformat(),
                entry.access_count,
                entry.similarity_threshold,
                entry.expiry_hours,
                entry.context_size,
                entry.token_savings
            ))
            self.db_connection.commit()
        except Exception as e:
            logging.error(f"Failed to save cache entry to database: {e}")
    
    def _generate_request_key(self, request_data: Dict[str, Any]) -> str:
        """Generate a semantic key for the request."""
        # Combine key fields that affect semantic meaning
        key_fields = [
            request_data.get("description", ""),
            request_data.get("project_type", ""),
            request_data.get("security_frameworks", []),
            request_data.get("agents", []),
            request_data.get("template_id", "")
        ]
        
        # Create text for embedding
        text_content = " ".join([
            str(field) for field in key_fields if field
        ])
        
        return text_content.strip()
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate semantic embedding for text."""
        if not self.embedding_model or not text.strip():
            return np.zeros(384)  # Default dimension for MiniLM
        
        try:
            embedding = self.embedding_model.encode([text])
            return embedding[0]
        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            return np.zeros(384)
    
    def _calculate_token_estimate(self, data: Dict[str, Any]) -> int:
        """Estimate token count for request/response data."""
        text_content = json.dumps(data, indent=2)
        # Rough estimate: 1 token ≈ 4 characters
        return len(text_content) // 4
    
    def get_cached_response(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response for semantically similar request."""
        
        with self.lock:
            # Generate semantic key and embedding
            semantic_key = self._generate_request_key(request_data)
            if not semantic_key:
                return None
            
            query_embedding = self._generate_embedding(semantic_key)
            
            # Find best matching cache entry
            best_match = None
            best_similarity = 0.0
            
            for entry in self.cache.values():
                if entry.is_expired():
                    continue
                
                similarity = entry.calculate_similarity(query_embedding)
                
                if (similarity >= entry.similarity_threshold and 
                    similarity > best_similarity):
                    best_match = entry
                    best_similarity = similarity
            
            if best_match:
                # Cache hit
                best_match.update_access()
                self.stats["hits"] += 1
                self.stats["token_savings"] += best_match.token_savings
                
                # Update database
                self._save_entry_to_db(best_match)
                
                logging.info(f"Cache HIT: similarity={best_similarity:.3f}, "
                           f"key={best_match.cache_key[:50]}")
                
                return best_match.response_data
            else:
                # Cache miss
                self.stats["misses"] += 1
                logging.info(f"Cache MISS for key: {semantic_key[:50]}")
                return None
    
    def cache_response(self, 
                      request_data: Dict[str, Any], 
                      response_data: Dict[str, Any],
                      similarity_threshold: float = None,
                      expiry_hours: int = None) -> str:
        """Cache a response with semantic indexing."""
        
        with self.lock:
            # Generate semantic key and embedding
            semantic_key = self._generate_request_key(request_data)
            if not semantic_key:
                return ""
            
            # Calculate token estimates
            request_tokens = self._calculate_token_estimate(request_data)
            response_tokens = self._calculate_token_estimate(response_data)
            total_tokens = request_tokens + response_tokens
            
            # Skip caching if too small
            if total_tokens < self.config.min_token_threshold:
                return ""
            
            # Skip caching if too large
            if total_tokens > self.config.max_context_size:
                logging.warning(f"Skipping cache: context too large ({total_tokens} tokens)")
                return ""
            
            # Generate cache key
            request_hash = hashlib.md5(
                json.dumps(request_data, sort_keys=True).encode()
            ).hexdigest()
            cache_key = f"{request_hash}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create cache entry
            entry = CacheEntry(
                cache_key=cache_key,
                request_hash=request_hash,
                semantic_embedding=self._generate_embedding(semantic_key),
                request_data=request_data.copy(),
                response_data=response_data.copy(),
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                similarity_threshold=similarity_threshold or self.config.similarity_threshold,
                expiry_hours=expiry_hours or self.config.default_expiry_hours,
                context_size=total_tokens,
                token_savings=total_tokens
            )
            
            # Add to cache
            self.cache[cache_key] = entry
            self.stats["entries_created"] += 1
            
            # Save to database
            self._save_entry_to_db(entry)
            
            # Cleanup if cache is too large
            self._cleanup_cache()
            
            logging.info(f"Cached response: key={cache_key[:50]}, tokens={total_tokens}")
            
            return cache_key
    
    def _cleanup_cache(self):
        """Clean up expired and excess cache entries."""
        current_time = datetime.utcnow()
        
        # Remove expired entries
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
            self.stats["entries_expired"] += len(expired_keys)
        
        # Remove excess entries if cache is too large
        if len(self.cache) > self.config.max_cache_size:
            # Sort by last accessed time and remove oldest
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            excess_count = len(self.cache) - self.config.max_cache_size
            for key, _ in sorted_entries[:excess_count]:
                del self.cache[key]
        
        # Clean database periodically
        if self.config.enable_persistence and self.db_connection:
            try:
                self.db_connection.execute("""
                    DELETE FROM cache_entries 
                    WHERE created_at < datetime('now', '-' || expiry_hours || ' hours')
                """)
                self.db_connection.commit()
            except Exception as e:
                logging.error(f"Database cleanup failed: {e}")
    
    def invalidate_cache(self, request_data: Optional[Dict[str, Any]] = None):
        """Invalidate cache entries matching request or all if None."""
        
        with self.lock:
            if request_data is None:
                # Clear all cache
                self.cache.clear()
                if self.config.enable_persistence and self.db_connection:
                    self.db_connection.execute("DELETE FROM cache_entries")
                    self.db_connection.commit()
                logging.info("Invalidated entire cache")
            else:
                # Invalidate specific entries
                request_hash = hashlib.md5(
                    json.dumps(request_data, sort_keys=True).encode()
                ).hexdigest()
                
                keys_to_remove = [
                    key for key, entry in self.cache.items()
                    if entry.request_hash == request_hash
                ]
                
                for key in keys_to_remove:
                    del self.cache[key]
                
                if self.config.enable_persistence and self.db_connection:
                    self.db_connection.execute(
                        "DELETE FROM cache_entries WHERE request_hash = ?",
                        (request_hash,)
                    )
                    self.db_connection.commit()
                
                logging.info(f"Invalidated {len(keys_to_remove)} cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics."""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "cache_size": len(self.cache),
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate_percent": round(hit_rate, 2),
                "token_savings": self.stats["token_savings"],
                "entries_created": self.stats["entries_created"],
                "entries_expired": self.stats["entries_expired"],
                "average_token_savings_per_hit": (
                    self.stats["token_savings"] // self.stats["hits"]
                    if self.stats["hits"] > 0 else 0
                )
            }
    
    def optimize_cache(self):
        """Optimize cache by removing low-value entries."""
        with self.lock:
            # Remove entries with low access counts and old timestamps
            threshold_time = datetime.utcnow() - timedelta(hours=48)
            
            keys_to_remove = [
                key for key, entry in self.cache.items()
                if (entry.access_count <= 1 and 
                    entry.created_at < threshold_time)
            ]
            
            for key in keys_to_remove:
                del self.cache[key]
            
            logging.info(f"Optimized cache: removed {len(keys_to_remove)} low-value entries")
            
            # Cleanup database
            self._cleanup_cache()
    
    def close(self):
        """Close database connection and cleanup."""
        if self.db_connection:
            # Save final statistics
            try:
                for stat_name, stat_value in self.stats.items():
                    self.db_connection.execute(
                        "INSERT OR REPLACE INTO cache_stats VALUES (?, ?)",
                        (stat_name, stat_value)
                    )
                self.db_connection.commit()
                self.db_connection.close()
            except Exception as e:
                logging.error(f"Failed to close database: {e}")


# Global cache instance
_cache_manager = None
_cache_lock = threading.Lock()

def get_cache_manager() -> SemanticCacheManager:
    """Get global cache manager instance."""
    global _cache_manager, _cache_lock
    
    with _cache_lock:
        if _cache_manager is None:
            _cache_manager = SemanticCacheManager()
        return _cache_manager

def cache_response(request_data: Dict[str, Any], 
                  response_data: Dict[str, Any],
                  **kwargs) -> str:
    """Convenience function to cache a response."""
    return get_cache_manager().cache_response(request_data, response_data, **kwargs)

def get_cached_response(request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convenience function to get cached response."""
    return get_cache_manager().get_cached_response(request_data)

def get_cache_stats() -> Dict[str, Any]:
    """Convenience function to get cache statistics."""
    return get_cache_manager().get_cache_stats()