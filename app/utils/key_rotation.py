"""
API Key rotation utility for load balancing and failover
"""
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
from loguru import logger


@dataclass
class KeyStats:
    """Statistics for API key usage"""
    key: str
    usage_count: int = 0
    error_count: int = 0
    last_used: Optional[datetime] = None
    last_error: Optional[datetime] = None
    is_blocked: bool = False
    block_until: Optional[datetime] = None


class KeyRotationManager:
    """
    Manages API key rotation with automatic failover and rate limiting
    
    Features:
    - Round-robin key selection
    - Automatic key blocking on repeated failures
    - Key health monitoring
    - Load balancing across keys
    """
    
    def __init__(
        self, 
        keys: List[str],
        max_errors_per_key: int = 3,
        block_duration_minutes: int = 5,
        enable_random_selection: bool = False
    ):
        if not keys:
            raise ValueError("At least one API key must be provided")
            
        self.keys = [key.strip() for key in keys if key.strip()]
        self.max_errors_per_key = max_errors_per_key
        self.block_duration = timedelta(minutes=block_duration_minutes)
        self.enable_random_selection = enable_random_selection
        
        # Initialize key statistics
        self.key_stats: Dict[str, KeyStats] = {
            key: KeyStats(key=key) for key in self.keys
        }
        
        # Current key index for round-robin
        self._current_index = 0
        self._lock = asyncio.Lock()

        logger.info(f"Initialized KeyRotationManager with {len(self.keys)} keys")
    
    async def get_next_key(self) -> Optional[str]:
        """
        Get the next available API key
        
        Returns:
            API key string or None if all keys are blocked
        """
        async with self._lock:
            # Clean up expired blocks
            await self._cleanup_expired_blocks()
            
            # Get available keys
            available_keys = [
                key for key, stats in self.key_stats.items() 
                if not stats.is_blocked
            ]
            
            if not available_keys:
                logger.warning("All API keys are currently blocked")
                return None
            
            # Select key based on strategy
            if self.enable_random_selection:
                selected_key = random.choice(available_keys)
            else:
                # Round-robin selection
                selected_key = self._get_round_robin_key(available_keys)
            
            # Update usage stats
            stats = self.key_stats[selected_key]
            stats.usage_count += 1
            stats.last_used = datetime.now()
            
            logger.debug(f"Selected API key: {selected_key[:8]}... (usage: {stats.usage_count})")
            return selected_key
    
    async def report_error(self, key: str, error: Exception) -> None:
        """
        Report an error for a specific key
        
        Args:
            key: The API key that encountered an error
            error: The exception that occurred
        """
        async with self._lock:
            if key not in self.key_stats:
                logger.warning(f"Attempted to report error for unknown key: {key[:8]}...")
                return
            
            stats = self.key_stats[key]
            stats.error_count += 1
            stats.last_error = datetime.now()
            
            logger.warning(f"Error reported for key {key[:8]}...: {error} (total errors: {stats.error_count})")
            
            # Block key if error threshold exceeded
            if stats.error_count >= self.max_errors_per_key:
                stats.is_blocked = True
                stats.block_until = datetime.now() + self.block_duration
                logger.error(f"Key {key[:8]}... blocked due to {stats.error_count} errors. Unblocked at {stats.block_until}")
    
    async def report_success(self, key: str) -> None:
        """
        Report successful usage of a key (resets error count)
        
        Args:
            key: The API key that was used successfully
        """
        async with self._lock:
            if key not in self.key_stats:
                return
                
            stats = self.key_stats[key]
            # Reset error count on successful usage
            stats.error_count = 0
            logger.debug(f"Success reported for key {key[:8]}... (errors reset)")
    
    async def get_key_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all keys
        
        Returns:
            Dictionary with key statistics
        """
        async with self._lock:
            return {
                key[:8] + "...": {
                    "usage_count": stats.usage_count,
                    "error_count": stats.error_count,
                    "last_used": stats.last_used.isoformat() if stats.last_used else None,
                    "last_error": stats.last_error.isoformat() if stats.last_error else None,
                    "is_blocked": stats.is_blocked,
                    "block_until": stats.block_until.isoformat() if stats.block_until else None
                }
                for key, stats in self.key_stats.items()
            }
    
    async def unblock_key(self, key: str) -> bool:
        """
        Manually unblock a key
        
        Args:
            key: The API key to unblock
            
        Returns:
            True if key was unblocked, False if key not found
        """
        async with self._lock:
            if key not in self.key_stats:
                return False
                
            stats = self.key_stats[key]
            stats.is_blocked = False
            stats.block_until = None
            stats.error_count = 0
            logger.info(f"Manually unblocked key {key[:8]}...")
            return True
    
    def _get_round_robin_key(self, available_keys: List[str]) -> str:
        """Get key using round-robin strategy"""
        # Find the next available key starting from current index
        original_keys_indices = {key: i for i, key in enumerate(self.keys)}
        
        # Sort available keys by their original index
        sorted_available = sorted(available_keys, key=lambda k: original_keys_indices[k])
        
        # Find current key position in available keys
        try:
            current_key = self.keys[self._current_index % len(self.keys)]
            if current_key in sorted_available:
                current_pos = sorted_available.index(current_key)
                selected_key = sorted_available[(current_pos + 1) % len(sorted_available)]
            else:
                selected_key = sorted_available[0]
        except (IndexError, ValueError):
            selected_key = sorted_available[0]
        
        # Update current index
        self._current_index = original_keys_indices[selected_key] + 1
        
        return selected_key
    
    async def _cleanup_expired_blocks(self) -> None:
        """Remove expired blocks from keys"""
        now = datetime.now()
        unblocked_count = 0
        
        for stats in self.key_stats.values():
            if stats.is_blocked and stats.block_until and now >= stats.block_until:
                stats.is_blocked = False
                stats.block_until = None
                stats.error_count = 0  # Reset error count
                unblocked_count += 1
                logger.info(f"Auto-unblocked key {stats.key[:8]}... after cooldown period")
        
        if unblocked_count > 0:
            logger.info(f"Auto-unblocked {unblocked_count} keys after cooldown") 