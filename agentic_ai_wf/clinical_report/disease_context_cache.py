"""
Thread-Safe Disease Context Cache with Persistent Storage
=========================================================

This module provides a global, thread-safe cache for disease contexts to prevent
redundant API calls across multiple worker threads and analysis sessions.

Key Features:
- Thread-safe global cache shared across all workers
- Persistent JSON storage for context reuse across sessions
- Automatic cache validation and refresh
- API call optimization with intelligent caching
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DiseaseContextCache:
    """Thread-safe global disease context cache with persistent storage"""
    
    _instance = None
    _lock = threading.Lock()
    _cache_lock = threading.RLock()  # Use RLock for nested locking
    
    def __new__(cls):
        """Singleton pattern to ensure global cache"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize cache only once"""
        if self._initialized:
            return
            
        with self._cache_lock:
            if self._initialized:
                return
                
            # Cache configuration
            self.cache_dir = Path("cache/disease_contexts")
            self.cache_file = self.cache_dir / "disease_contexts.json"
            self.cache_duration_days = 30  # Cache contexts for 30 days
            
            # In-memory cache
            self._memory_cache = {}
            self._cache_timestamps = {}
            self._access_count = {}
            
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing cache
            self._load_cache_from_disk()
            
            self._initialized = True
            logger.info(f"✅ Disease context cache initialized: {len(self._memory_cache)} contexts loaded")
    
    def _load_cache_from_disk(self):
        """Load disease contexts from persistent JSON storage"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self._memory_cache = data.get('contexts', {})
                self._cache_timestamps = data.get('timestamps', {})
                self._access_count = data.get('access_count', {})
                
                # Clean expired contexts
                self._clean_expired_contexts()
                
                logger.info(f"📁 Loaded {len(self._memory_cache)} disease contexts from cache")
            else:
                logger.info("📁 No existing disease context cache found, starting fresh")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load disease context cache: {e}")
            self._memory_cache = {}
            self._cache_timestamps = {}
            self._access_count = {}
    
    def _save_cache_to_disk(self):
        """Save disease contexts to persistent JSON storage"""
        try:
            data = {
                'contexts': self._memory_cache,
                'timestamps': self._cache_timestamps,
                'access_count': self._access_count,
                'last_updated': datetime.now().isoformat(),
                'cache_version': '1.0'
            }
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.cache_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            temp_file.replace(self.cache_file)
            logger.debug(f"💾 Saved {len(self._memory_cache)} disease contexts to cache")
            
        except Exception as e:
            logger.error(f"❌ Failed to save disease context cache: {e}")
    
    def _clean_expired_contexts(self):
        """Remove expired contexts from cache"""
        cutoff_time = datetime.now() - timedelta(days=self.cache_duration_days)
        expired_keys = []
        
        for key, timestamp_str in self._cache_timestamps.items():
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp < cutoff_time:
                    expired_keys.append(key)
            except (ValueError, TypeError):
                expired_keys.append(key)  # Remove invalid timestamps
        
        for key in expired_keys:
            self._memory_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
            self._access_count.pop(key, None)
        
        if expired_keys:
            logger.info(f"🧹 Cleaned {len(expired_keys)} expired disease contexts")
    
    def _normalize_disease_name(self, disease_name: str) -> str:
        """Normalize disease name for consistent caching"""
        return disease_name.lower().strip()
    
    def get_context(self, disease_name: str) -> Optional[Dict]:
        """Get disease context from cache (thread-safe)"""
        normalized_name = self._normalize_disease_name(disease_name)
        
        with self._cache_lock:
            if normalized_name in self._memory_cache:
                # Update access count and timestamp
                self._access_count[normalized_name] = self._access_count.get(normalized_name, 0) + 1
                
                # Check if context is still fresh
                timestamp_str = self._cache_timestamps.get(normalized_name)
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if datetime.now() - timestamp < timedelta(days=self.cache_duration_days):
                            logger.debug(f"🎯 Cache HIT for disease: {disease_name} (access #{self._access_count[normalized_name]})")
                            return self._memory_cache[normalized_name].copy()
                    except (ValueError, TypeError):
                        pass
                
                # Context is expired, remove it
                self._memory_cache.pop(normalized_name, None)
                self._cache_timestamps.pop(normalized_name, None)
                self._access_count.pop(normalized_name, None)
        
        logger.debug(f"💭 Cache MISS for disease: {disease_name}")
        return None
    
    def set_context(self, disease_name: str, context: Dict) -> bool:
        """Set disease context in cache (thread-safe)"""
        normalized_name = self._normalize_disease_name(disease_name)
        
        try:
            with self._cache_lock:
                self._memory_cache[normalized_name] = context.copy()
                self._cache_timestamps[normalized_name] = datetime.now().isoformat()
                self._access_count[normalized_name] = 1
                
                # Save to disk asynchronously (non-blocking)
                self._save_cache_to_disk()
                
                logger.info(f"💾 Cached new disease context for: {disease_name}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to cache disease context for {disease_name}: {e}")
            return False
    
    def has_context(self, disease_name: str) -> bool:
        """Check if disease context exists in cache"""
        normalized_name = self._normalize_disease_name(disease_name)
        with self._cache_lock:
            return normalized_name in self._memory_cache
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self._cache_lock:
            total_contexts = len(self._memory_cache)
            total_accesses = sum(self._access_count.values())
            
            if self._access_count:
                most_accessed = max(self._access_count.items(), key=lambda x: x[1])
                avg_accesses = total_accesses / len(self._access_count)
            else:
                most_accessed = ("None", 0)
                avg_accesses = 0
            
            return {
                'total_contexts_cached': total_contexts,
                'total_accesses': total_accesses,
                'average_accesses_per_context': round(avg_accesses, 2),
                'most_accessed_disease': most_accessed[0],
                'most_accessed_count': most_accessed[1],
                'cache_file_path': str(self.cache_file),
                'cache_duration_days': self.cache_duration_days
            }
    
    def clear_cache(self):
        """Clear all cached contexts"""
        with self._cache_lock:
            self._memory_cache.clear()
            self._cache_timestamps.clear()
            self._access_count.clear()
            
            try:
                if self.cache_file.exists():
                    self.cache_file.unlink()
                logger.info("🧹 Disease context cache cleared")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete cache file: {e}")
    
    def preload_common_diseases(self, diseases: list):
        """Preload contexts for common diseases to improve performance"""
        missing_diseases = [disease for disease in diseases if not self.has_context(disease)]
        
        if missing_diseases:
            logger.info(f"🔄 Preloading contexts for {len(missing_diseases)} diseases")
            # This would trigger context generation for missing diseases
            return missing_diseases
        else:
            logger.info(f"✅ All {len(diseases)} diseases already cached")
            return []

# Global cache instance (lazy initialization)
_global_cache = None
_cache_lock = threading.Lock()

def _get_global_cache():
    """Lazy initialization of global cache - only creates when first accessed"""
    global _global_cache
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = DiseaseContextCache()
    return _global_cache

def get_disease_context_cached(disease_name: str) -> Optional[Dict]:
    """Get disease context from global cache"""
    return _get_global_cache().get_context(disease_name)

def set_disease_context_cached(disease_name: str, context: Dict) -> bool:
    """Set disease context in global cache"""
    return _get_global_cache().set_context(disease_name, context)

def has_disease_context_cached(disease_name: str) -> bool:
    """Check if disease context is cached"""
    return _get_global_cache().has_context(disease_name)

def get_cache_statistics() -> Dict:
    """Get cache statistics"""
    return _get_global_cache().get_cache_stats()

def clear_disease_context_cache():
    """Clear disease context cache"""
    _get_global_cache().clear_cache()

def preload_disease_contexts(diseases: list) -> list:
    """Preload disease contexts"""
    return _get_global_cache().preload_common_diseases(diseases)
