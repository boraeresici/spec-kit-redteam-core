#!/usr/bin/env python3
"""
Intelligent caching system for collaborative AI specification generation.
Reduces token usage and costs through smart response caching.
"""

import json
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import platformdirs

from .agents.base_agent import AgentResponse
from .token_tracker import TokenTracker


@dataclass
class CacheEntry:
    """A cached agent response with metadata"""
    agent_role: str
    input_hash: str
    response: AgentResponse
    timestamp: datetime
    usage_count: int = 0
    input_preview: str = ""
    tags: List[str] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['response'] = self.response.to_dict()
        data['timestamp'] = self.timestamp.isoformat()
        data['tags'] = self.tags or []
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheEntry':
        data['response'] = AgentResponse.from_dict(data['response'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['tags'] = data.get('tags', [])
        return cls(**data)


class SpecCache:
    """Intelligent caching system for agent responses"""
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 ttl_hours: int = 24,
                 max_entries: int = 1000):
        
        self.cache_dir = cache_dir or Path(platformdirs.user_cache_dir("specify-cli")) / "agent_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.ttl = timedelta(hours=ttl_hours)
        self.max_entries = max_entries
        
        # In-memory cache for fast access
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._load_index()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0,
            'evictions': 0,
            'tokens_saved': 0
        }
    
    def _load_index(self):
        """Load cache index from disk"""
        index_file = self.cache_dir / "cache_index.json"
        
        if index_file.exists():
            try:
                with open(index_file) as f:
                    index_data = json.load(f)
                
                for cache_key, entry_data in index_data.items():
                    try:
                        entry = CacheEntry.from_dict(entry_data)
                        # Skip expired entries
                        if datetime.now() - entry.timestamp <= self.ttl:
                            self._memory_cache[cache_key] = entry
                    except Exception:
                        continue  # Skip corrupted entries
                        
            except (json.JSONDecodeError, FileNotFoundError):
                pass  # Start with empty cache
    
    def _save_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / "cache_index.json"
        
        # Clean up expired entries before saving
        self._cleanup_expired()
        
        index_data = {}
        for cache_key, entry in self._memory_cache.items():
            index_data[cache_key] = entry.to_dict()
        
        try:
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
        except Exception:
            pass  # Fail silently to not disrupt main flow
    
    def _generate_cache_key(self, agent_role: str, input_text: str, context: Dict = None) -> str:
        """Generate cache key from input"""
        # Normalize input for better cache hits
        normalized_input = input_text.lower().strip()
        
        # Include relevant context in key
        context_str = ""
        if context:
            # Only include stable context elements
            stable_context = {
                k: v for k, v in context.items() 
                if k in ['operation', 'complexity', 'agent_role']
            }
            context_str = json.dumps(stable_context, sort_keys=True)
        
        combined = f"{agent_role}:{normalized_input}:{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _extract_semantic_tags(self, input_text: str) -> List[str]:
        """Extract semantic tags from input for better cache matching"""
        tags = []
        
        # Domain tags
        if any(word in input_text.lower() for word in ['chat', 'message', 'communication']):
            tags.append('communication')
        if any(word in input_text.lower() for word in ['task', 'project', 'management']):
            tags.append('project_management')
        if any(word in input_text.lower() for word in ['user', 'auth', 'login', 'permission']):
            tags.append('user_management')
        if any(word in input_text.lower() for word in ['api', 'rest', 'graphql', 'endpoint']):
            tags.append('api')
        if any(word in input_text.lower() for word in ['database', 'storage', 'data']):
            tags.append('data_storage')
        if any(word in input_text.lower() for word in ['real-time', 'websocket', 'live']):
            tags.append('realtime')
        
        # Complexity tags
        if len(input_text) > 500:
            tags.append('complex')
        elif len(input_text) < 100:
            tags.append('simple')
        else:
            tags.append('medium')
        
        return tags
    
    def get_cached_response(self, 
                           agent_role: str,
                           input_text: str,
                           context: Dict = None) -> Optional[AgentResponse]:
        """Retrieve cached response if available and fresh"""
        
        cache_key = self._generate_cache_key(agent_role, input_text, context)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]
            
            # Check if expired
            if datetime.now() - entry.timestamp > self.ttl:
                del self._memory_cache[cache_key]
                self.stats['misses'] += 1
                return None
            
            # Update usage stats
            entry.usage_count += 1
            self.stats['hits'] += 1
            self.stats['tokens_saved'] += entry.response.input_tokens + entry.response.output_tokens
            
            return entry.response
        
        # Try semantic matching for similar inputs
        similar_response = self._find_similar_response(agent_role, input_text)
        if similar_response:
            self.stats['hits'] += 1
            return similar_response
        
        self.stats['misses'] += 1
        return None
    
    def _find_similar_response(self, agent_role: str, input_text: str) -> Optional[AgentResponse]:
        """Find semantically similar cached responses"""
        
        input_tags = set(self._extract_semantic_tags(input_text))
        input_words = set(input_text.lower().split())
        
        best_match = None
        best_similarity = 0.0
        
        for entry in self._memory_cache.values():
            if entry.agent_role != agent_role:
                continue
                
            # Check tag overlap
            entry_tags = set(entry.tags or [])
            tag_similarity = len(input_tags & entry_tags) / max(len(input_tags | entry_tags), 1)
            
            # Check word overlap
            entry_words = set(entry.input_preview.lower().split())
            word_similarity = len(input_words & entry_words) / max(len(input_words | entry_words), 1)
            
            # Combined similarity score
            combined_similarity = (tag_similarity * 0.6) + (word_similarity * 0.4)
            
            # Only consider high similarity matches
            if combined_similarity > 0.7 and combined_similarity > best_similarity:
                best_similarity = combined_similarity
                best_match = entry.response
        
        return best_match
    
    def cache_response(self, 
                      agent_role: str,
                      input_text: str,
                      response: AgentResponse,
                      context: Dict = None):
        """Cache an agent response"""
        
        cache_key = self._generate_cache_key(agent_role, input_text, context)
        
        entry = CacheEntry(
            agent_role=agent_role,
            input_hash=cache_key,
            response=response,
            timestamp=datetime.now(),
            input_preview=input_text[:200],  # Store preview for similarity matching
            tags=self._extract_semantic_tags(input_text)
        )
        
        self._memory_cache[cache_key] = entry
        self.stats['saves'] += 1
        
        # Evict old entries if cache is full
        if len(self._memory_cache) > self.max_entries:
            self._evict_oldest_entries()
        
        # Periodically save to disk
        if self.stats['saves'] % 10 == 0:
            self._save_index()
    
    def _evict_oldest_entries(self):
        """Evict oldest entries when cache is full"""
        # Sort by timestamp and usage count
        entries = list(self._memory_cache.items())
        entries.sort(key=lambda x: (x[1].timestamp, x[1].usage_count))
        
        # Remove oldest 10% of entries
        evict_count = max(1, len(entries) // 10)
        
        for i in range(evict_count):
            cache_key = entries[i][0]
            del self._memory_cache[cache_key]
            self.stats['evictions'] += 1
    
    def _cleanup_expired(self):
        """Remove expired entries from memory cache"""
        expired_keys = []
        now = datetime.now()
        
        for cache_key, entry in self._memory_cache.items():
            if now - entry.timestamp > self.ttl:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self._memory_cache[key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / max(total_requests, 1)) * 100
        
        return {
            'hit_rate_percentage': hit_rate,
            'total_hits': self.stats['hits'],
            'total_misses': self.stats['misses'],
            'total_saves': self.stats['saves'],
            'total_evictions': self.stats['evictions'],
            'tokens_saved': self.stats['tokens_saved'],
            'cache_size': len(self._memory_cache),
            'estimated_cost_savings': self._estimate_cost_savings()
        }
    
    def _estimate_cost_savings(self) -> float:
        """Estimate cost savings from cache hits"""
        # Rough estimate: $0.50 per 1000 tokens saved
        return (self.stats['tokens_saved'] / 1000) * 0.50
    
    def warm_cache_from_history(self, token_tracker: TokenTracker):
        """Pre-warm cache with common patterns from usage history"""
        # This would analyze historical usage patterns and pre-cache common requests
        # For now, just a placeholder
        pass
    
    def clear_cache(self):
        """Clear all cached entries"""
        self._memory_cache.clear()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0,
            'evictions': 0,
            'tokens_saved': 0
        }
        
        # Remove cache files
        try:
            index_file = self.cache_dir / "cache_index.json"
            if index_file.exists():
                index_file.unlink()
        except Exception:
            pass
    
    def export_cache_analysis(self) -> Dict[str, Any]:
        """Export detailed cache analysis for optimization"""
        
        # Analyze cache patterns
        agent_usage = {}
        tag_popularity = {}
        usage_patterns = {}
        
        for entry in self._memory_cache.values():
            # Agent usage
            if entry.agent_role not in agent_usage:
                agent_usage[entry.agent_role] = {'count': 0, 'usage_total': 0}
            agent_usage[entry.agent_role]['count'] += 1
            agent_usage[entry.agent_role]['usage_total'] += entry.usage_count
            
            # Tag popularity
            for tag in entry.tags or []:
                tag_popularity[tag] = tag_popularity.get(tag, 0) + 1
            
            # Usage patterns
            age_days = (datetime.now() - entry.timestamp).days
            age_bucket = f"{age_days//7}w" if age_days >= 7 else f"{age_days}d"
            usage_patterns[age_bucket] = usage_patterns.get(age_bucket, 0) + entry.usage_count
        
        return {
            'cache_stats': self.get_cache_stats(),
            'agent_usage_patterns': agent_usage,
            'popular_tags': dict(sorted(tag_popularity.items(), key=lambda x: x[1], reverse=True)[:10]),
            'usage_by_age': usage_patterns,
            'optimization_suggestions': self._generate_optimization_suggestions(agent_usage, tag_popularity)
        }
    
    def _generate_optimization_suggestions(self, 
                                         agent_usage: Dict, 
                                         tag_popularity: Dict) -> List[str]:
        """Generate suggestions for cache optimization"""
        suggestions = []
        
        # Suggest increasing TTL for frequently used patterns
        high_usage_agents = [
            agent for agent, stats in agent_usage.items()
            if stats['usage_total'] > 5
        ]
        
        if high_usage_agents:
            suggestions.append(f"Consider increasing TTL for high-usage agents: {', '.join(high_usage_agents)}")
        
        # Suggest pre-warming for popular tags
        popular_tags = [tag for tag, count in tag_popularity.items() if count > 3]
        if popular_tags:
            suggestions.append(f"Consider pre-warming cache for popular domains: {', '.join(popular_tags[:3])}")
        
        # Cache size recommendations
        cache_utilization = len(self._memory_cache) / self.max_entries
        if cache_utilization > 0.8:
            suggestions.append("Consider increasing cache size limit for better retention")
        elif cache_utilization < 0.3:
            suggestions.append("Cache size could be reduced to save memory")
        
        return suggestions
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._save_index()


class CacheOptimizer:
    """Optimizes caching strategies based on usage patterns"""
    
    def __init__(self, cache: SpecCache):
        self.cache = cache
    
    def optimize_cache_settings(self) -> Dict[str, Any]:
        """Analyze usage and suggest optimal cache settings"""
        
        analysis = self.cache.export_cache_analysis()
        stats = analysis['cache_stats']
        
        recommendations = {
            'current_settings': {
                'ttl_hours': self.cache.ttl.total_seconds() / 3600,
                'max_entries': self.cache.max_entries,
                'hit_rate': stats['hit_rate_percentage']
            },
            'optimized_settings': {},
            'expected_improvements': {}
        }
        
        # TTL optimization
        if stats['hit_rate_percentage'] < 50:
            # Low hit rate might indicate TTL is too short
            new_ttl = min(72, self.cache.ttl.total_seconds() / 3600 * 1.5)
            recommendations['optimized_settings']['ttl_hours'] = new_ttl
            recommendations['expected_improvements']['hit_rate_increase'] = "10-20%"
        
        # Cache size optimization
        if stats['total_evictions'] > stats['total_hits'] * 0.1:
            # Too many evictions relative to hits
            new_size = min(2000, self.cache.max_entries * 1.5)
            recommendations['optimized_settings']['max_entries'] = new_size
            recommendations['expected_improvements']['eviction_reduction'] = "50%"
        
        return recommendations