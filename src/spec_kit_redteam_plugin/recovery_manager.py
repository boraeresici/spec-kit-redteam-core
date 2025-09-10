#!/usr/bin/env python3
"""
Error Recovery and Graceful Fallback System for Spec-Kit RED TEAM Plugin

Handles network failures, API rate limits, and other recoverable errors with
automatic retry logic and graceful degradation.
"""

import time
import asyncio
import random
from typing import Dict, List, Optional, Any, Callable, TypeVar, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

T = TypeVar('T')


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY_WITH_BACKOFF = "retry_backoff"
    RETRY_WITH_REDUCED_REQUEST = "retry_reduced"
    FALLBACK_TO_CACHED = "fallback_cache"
    FALLBACK_TO_SIMPLIFIED = "fallback_simplified"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ABORT_WITH_PARTIAL = "abort_partial"


class ErrorType(Enum):
    """Categorization of recoverable errors"""
    NETWORK_TIMEOUT = "network_timeout"
    NETWORK_CONNECTION = "network_connection"
    API_RATE_LIMIT = "api_rate_limit"
    API_QUOTA_EXCEEDED = "api_quota_exceeded"
    API_SERVER_ERROR = "api_server_error"
    MEMORY_LIMIT = "memory_limit"
    BUDGET_LIMIT = "budget_limit"
    PROCESSING_TIMEOUT = "processing_timeout"


@dataclass
class RecoveryConfig:
    """Configuration for recovery attempts"""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True
    timeout: float = 120.0


@dataclass
class RecoveryAttempt:
    """Information about a recovery attempt"""
    attempt_number: int
    strategy: RecoveryStrategy
    error_type: ErrorType
    timestamp: float
    success: bool
    details: Optional[str] = None


class RecoveryManager:
    """Manages error recovery with multiple fallback strategies"""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.recovery_history: List[RecoveryAttempt] = []
        
        # Default recovery strategies by error type
        self.recovery_strategies = {
            ErrorType.NETWORK_TIMEOUT: [
                RecoveryStrategy.RETRY_WITH_BACKOFF,
                RecoveryStrategy.RETRY_WITH_REDUCED_REQUEST,
                RecoveryStrategy.FALLBACK_TO_CACHED
            ],
            ErrorType.NETWORK_CONNECTION: [
                RecoveryStrategy.RETRY_WITH_BACKOFF,
                RecoveryStrategy.FALLBACK_TO_CACHED,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            ErrorType.API_RATE_LIMIT: [
                RecoveryStrategy.RETRY_WITH_BACKOFF,
                RecoveryStrategy.FALLBACK_TO_CACHED,
                RecoveryStrategy.RETRY_WITH_REDUCED_REQUEST
            ],
            ErrorType.API_QUOTA_EXCEEDED: [
                RecoveryStrategy.FALLBACK_TO_CACHED,
                RecoveryStrategy.FALLBACK_TO_SIMPLIFIED,
                RecoveryStrategy.ABORT_WITH_PARTIAL
            ],
            ErrorType.API_SERVER_ERROR: [
                RecoveryStrategy.RETRY_WITH_BACKOFF,
                RecoveryStrategy.FALLBACK_TO_CACHED,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            ErrorType.MEMORY_LIMIT: [
                RecoveryStrategy.RETRY_WITH_REDUCED_REQUEST,
                RecoveryStrategy.FALLBACK_TO_SIMPLIFIED,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            ErrorType.BUDGET_LIMIT: [
                RecoveryStrategy.FALLBACK_TO_CACHED,
                RecoveryStrategy.FALLBACK_TO_SIMPLIFIED,
                RecoveryStrategy.ABORT_WITH_PARTIAL
            ],
            ErrorType.PROCESSING_TIMEOUT: [
                RecoveryStrategy.RETRY_WITH_REDUCED_REQUEST,
                RecoveryStrategy.FALLBACK_TO_SIMPLIFIED,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ]
        }
        
        # Default configurations by strategy
        self.strategy_configs = {
            RecoveryStrategy.RETRY_WITH_BACKOFF: RecoveryConfig(
                max_retries=3, base_delay=1.0, backoff_factor=2.0
            ),
            RecoveryStrategy.RETRY_WITH_REDUCED_REQUEST: RecoveryConfig(
                max_retries=2, base_delay=0.5, backoff_factor=1.5
            ),
            RecoveryStrategy.FALLBACK_TO_CACHED: RecoveryConfig(
                max_retries=1, base_delay=0.1
            ),
            RecoveryStrategy.FALLBACK_TO_SIMPLIFIED: RecoveryConfig(
                max_retries=2, base_delay=0.5
            ),
            RecoveryStrategy.GRACEFUL_DEGRADATION: RecoveryConfig(
                max_retries=1, base_delay=0.1
            ),
            RecoveryStrategy.ABORT_WITH_PARTIAL: RecoveryConfig(
                max_retries=1, base_delay=0.0
            )
        }
    
    def classify_error(self, error: Exception) -> ErrorType:
        """Classify an error to determine recovery strategy"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Network-related errors
        if any(keyword in error_str for keyword in ["timeout", "timed out"]):
            return ErrorType.NETWORK_TIMEOUT
        
        if any(keyword in error_str for keyword in ["connection", "unreachable", "network"]):
            return ErrorType.NETWORK_CONNECTION
        
        # API-related errors
        if "rate limit" in error_str or "429" in error_str:
            return ErrorType.API_RATE_LIMIT
        
        if "quota" in error_str or "limit exceeded" in error_str:
            return ErrorType.API_QUOTA_EXCEEDED
        
        if any(code in error_str for code in ["500", "502", "503", "504"]):
            return ErrorType.API_SERVER_ERROR
        
        # Resource-related errors
        if "memory" in error_str or "out of memory" in error_str:
            return ErrorType.MEMORY_LIMIT
        
        if "budget" in error_str or "cost" in error_str:
            return ErrorType.BUDGET_LIMIT
        
        # Default to processing timeout for unknown errors
        return ErrorType.PROCESSING_TIMEOUT
    
    async def recover_with_strategy(
        self,
        strategy: RecoveryStrategy,
        operation: Callable[..., T],
        error_context: Dict[str, Any],
        config: Optional[RecoveryConfig] = None,
        *args,
        **kwargs
    ) -> Optional[T]:
        """Execute recovery strategy for a failed operation"""
        
        if config is None:
            config = self.strategy_configs.get(strategy, RecoveryConfig())
        
        recovery_attempt = RecoveryAttempt(
            attempt_number=1,
            strategy=strategy,
            error_type=error_context.get('error_type', ErrorType.PROCESSING_TIMEOUT),
            timestamp=time.time(),
            success=False
        )
        
        try:
            if strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
                return await self._retry_with_backoff(operation, config, *args, **kwargs)
            
            elif strategy == RecoveryStrategy.RETRY_WITH_REDUCED_REQUEST:
                return await self._retry_with_reduced_request(
                    operation, error_context, config, *args, **kwargs
                )
            
            elif strategy == RecoveryStrategy.FALLBACK_TO_CACHED:
                return await self._fallback_to_cached(
                    operation, error_context, *args, **kwargs
                )
            
            elif strategy == RecoveryStrategy.FALLBACK_TO_SIMPLIFIED:
                return await self._fallback_to_simplified(
                    operation, error_context, *args, **kwargs
                )
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return await self._graceful_degradation(
                    operation, error_context, *args, **kwargs
                )
            
            elif strategy == RecoveryStrategy.ABORT_WITH_PARTIAL:
                return await self._abort_with_partial(
                    error_context, *args, **kwargs
                )
            
            else:
                raise ValueError(f"Unknown recovery strategy: {strategy}")
        
        except Exception as e:
            recovery_attempt.success = False
            recovery_attempt.details = str(e)
            self.recovery_history.append(recovery_attempt)
            raise
        
        else:
            recovery_attempt.success = True
            self.recovery_history.append(recovery_attempt)
    
    async def _retry_with_backoff(
        self,
        operation: Callable[..., T],
        config: RecoveryConfig,
        *args,
        **kwargs
    ) -> T:
        """Retry operation with exponential backoff"""
        
        last_exception = None
        
        for attempt in range(config.max_retries):
            try:
                if attempt > 0:
                    # Calculate delay with backoff
                    delay = min(
                        config.base_delay * (config.backoff_factor ** (attempt - 1)),
                        config.max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    self.console.print(
                        f"⏳ Retrying in {delay:.1f}s (attempt {attempt + 1}/{config.max_retries})...",
                        style="yellow"
                    )
                    await asyncio.sleep(delay)
                
                # Try the operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                if attempt > 0:
                    self.console.print(
                        f"✅ Operation succeeded on attempt {attempt + 1}",
                        style="green"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                if attempt == config.max_retries - 1:
                    # Last attempt failed
                    break
                
                self.console.print(
                    f"❌ Attempt {attempt + 1} failed: {str(e)[:100]}...",
                    style="red"
                )
        
        # All retries failed
        raise last_exception
    
    async def _retry_with_reduced_request(
        self,
        operation: Callable[..., T],
        error_context: Dict[str, Any],
        config: RecoveryConfig,
        *args,
        **kwargs
    ) -> T:
        """Retry with reduced request size/complexity"""
        
        # Modify request parameters to reduce load
        modified_kwargs = kwargs.copy()
        
        # Reduce batch sizes
        if 'batch_size' in modified_kwargs:
            modified_kwargs['batch_size'] = max(1, modified_kwargs['batch_size'] // 2)
        
        # Reduce complexity
        if 'complexity' in modified_kwargs:
            complexity_map = {'complex': 'medium', 'medium': 'simple'}
            modified_kwargs['complexity'] = complexity_map.get(
                modified_kwargs['complexity'], 'simple'
            )
        
        # Reduce agent count
        if 'agents' in modified_kwargs and isinstance(modified_kwargs['agents'], list):
            if len(modified_kwargs['agents']) > 2:
                modified_kwargs['agents'] = modified_kwargs['agents'][:2]
        
        # Reduce timeout
        if 'timeout' in modified_kwargs:
            modified_kwargs['timeout'] = modified_kwargs['timeout'] * 0.8
        
        self.console.print(
            "🔽 Retrying with reduced request complexity...",
            style="cyan"
        )
        
        return await self._retry_with_backoff(operation, config, *args, **modified_kwargs)
    
    async def _fallback_to_cached(
        self,
        operation: Callable[..., T],
        error_context: Dict[str, Any],
        *args,
        **kwargs
    ) -> Optional[T]:
        """Try to use cached results if available"""
        
        self.console.print("💾 Attempting to use cached results...", style="cyan")
        
        # Try to enable caching if not already enabled
        modified_kwargs = kwargs.copy()
        modified_kwargs['use_cache'] = True
        modified_kwargs['cache_only'] = True  # Only use cache, don't make new requests
        
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **modified_kwargs)
            else:
                result = operation(*args, **modified_kwargs)
            
            if result:
                self.console.print("✅ Used cached results successfully", style="green")
                return result
            else:
                self.console.print("⚠️ No cached results available", style="yellow")
                return None
        
        except Exception as e:
            self.console.print(f"❌ Cache fallback failed: {e}", style="red")
            return None
    
    async def _fallback_to_simplified(
        self,
        operation: Callable[..., T], 
        error_context: Dict[str, Any],
        *args,
        **kwargs
    ) -> Optional[T]:
        """Fallback to a simplified version of the operation"""
        
        self.console.print("📉 Falling back to simplified operation...", style="cyan")
        
        # Create simplified parameters
        simplified_kwargs = kwargs.copy()
        
        # Use only essential agents
        if 'agents' in simplified_kwargs:
            simplified_kwargs['agents'] = ['pm']  # Minimal viable agent set
        
        # Use simple complexity
        simplified_kwargs['complexity'] = 'simple'
        
        # Reduce budget if applicable
        if 'budget' in simplified_kwargs:
            simplified_kwargs['budget'] = min(
                simplified_kwargs['budget'], 10.0
            )  # Conservative budget
        
        # Disable advanced features
        simplified_kwargs['parallel'] = False
        simplified_kwargs['advanced_analysis'] = False
        
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **simplified_kwargs)
            else:
                result = operation(*args, **simplified_kwargs)
            
            if result:
                self.console.print(
                    "✅ Simplified operation completed successfully",
                    style="green"
                )
                self.console.print(
                    "ℹ️ Note: Results may be less comprehensive due to simplified mode",
                    style="dim"
                )
                return result
            
        except Exception as e:
            self.console.print(f"❌ Simplified fallback failed: {e}", style="red")
        
        return None
    
    async def _graceful_degradation(
        self,
        operation: Callable[..., T],
        error_context: Dict[str, Any],
        *args,
        **kwargs
    ) -> Optional[T]:
        """Gracefully degrade functionality while providing partial results"""
        
        self.console.print("⬇️ Graceful degradation in progress...", style="cyan")
        
        # Try to extract any partial results from error context
        partial_results = error_context.get('partial_results')
        
        if partial_results:
            self.console.print("📋 Partial results available:", style="yellow")
            
            # Display what we have
            if isinstance(partial_results, dict):
                for key, value in partial_results.items():
                    if value:
                        self.console.print(f"  ✅ {key}: Available", style="green")
                    else:
                        self.console.print(f"  ❌ {key}: Unavailable", style="red")
            
            # Return partial results wrapped in a suitable format
            return partial_results
        
        # Try to provide a minimal working result
        minimal_result = {
            'status': 'degraded',
            'message': 'Operation completed with degraded functionality',
            'available_features': error_context.get('available_features', []),
            'unavailable_features': error_context.get('unavailable_features', []),
            'recommendation': 'Try again later when services are fully available'
        }
        
        return minimal_result
    
    async def _abort_with_partial(
        self,
        error_context: Dict[str, Any],
        *args,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Abort operation but provide partial results and guidance"""
        
        self.console.print("🛑 Operation aborted - providing partial results", style="yellow")
        
        partial_info = {
            'status': 'aborted',
            'error_summary': error_context.get('error_message', 'Unknown error'),
            'completed_steps': error_context.get('completed_steps', []),
            'failed_step': error_context.get('failed_step', 'Unknown'),
            'partial_results': error_context.get('partial_results', {}),
            'recovery_suggestions': [
                'Check your internet connection',
                'Verify API credentials and quotas',
                'Try with reduced complexity or fewer agents',
                'Try again later when services are available'
            ],
            'next_actions': [
                'Run: specify collab doctor  # Diagnose issues',
                'Run: specify collab version --check-updates  # Check for fixes',
                'Contact support if issue persists'
            ]
        }
        
        return partial_info
    
    async def attempt_recovery(
        self,
        operation: Callable[..., T],
        error: Exception,
        context: Dict[str, Any],
        *args,
        **kwargs
    ) -> Optional[T]:
        """
        Main recovery orchestration method
        
        Attempts multiple recovery strategies based on error type
        """
        
        error_type = self.classify_error(error)
        strategies = self.recovery_strategies.get(error_type, [RecoveryStrategy.RETRY_WITH_BACKOFF])
        
        error_context = {
            'error_type': error_type,
            'original_error': error,
            'error_message': str(error),
            **context
        }
        
        self.console.print(
            f"\n🔄 Starting recovery process for {error_type.value}...",
            style="cyan"
        )
        
        for i, strategy in enumerate(strategies):
            try:
                self.console.print(
                    f"📋 Trying strategy {i+1}/{len(strategies)}: {strategy.value}",
                    style="blue"
                )
                
                result = await self.recover_with_strategy(
                    strategy, operation, error_context, None, *args, **kwargs
                )
                
                if result is not None:
                    self.console.print(
                        f"✅ Recovery successful with strategy: {strategy.value}",
                        style="bold green"
                    )
                    return result
                
            except Exception as recovery_error:
                self.console.print(
                    f"❌ Strategy {strategy.value} failed: {str(recovery_error)[:100]}...",
                    style="red"
                )
                
                # Continue to next strategy
                continue
        
        # All recovery strategies failed
        self.console.print(
            "💥 All recovery strategies failed",
            style="bold red"
        )
        
        return None
    
    def show_recovery_stats(self):
        """Show recovery attempt statistics"""
        if not self.recovery_history:
            self.console.print("📊 No recovery attempts recorded", style="green")
            return
        
        from collections import Counter
        
        # Strategy success rates
        strategy_stats = Counter()
        strategy_success = Counter()
        
        for attempt in self.recovery_history:
            strategy_stats[attempt.strategy] += 1
            if attempt.success:
                strategy_success[attempt.strategy] += 1
        
        stats_table = Table(title="🔄 Recovery Statistics")
        stats_table.add_column("Strategy", style="cyan")
        stats_table.add_column("Attempts", justify="right")
        stats_table.add_column("Success Rate", justify="right", style="green")
        
        for strategy in RecoveryStrategy:
            attempts = strategy_stats[strategy]
            successes = strategy_success[strategy]
            
            if attempts > 0:
                success_rate = (successes / attempts) * 100
                stats_table.add_row(
                    strategy.value,
                    str(attempts),
                    f"{success_rate:.1f}%"
                )
        
        self.console.print(stats_table)


# Global recovery manager instance
recovery_manager = RecoveryManager()


def with_recovery(
    operation_name: str = "operation",
    context: Optional[Dict[str, Any]] = None
):
    """Decorator to add automatic recovery to functions"""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            except Exception as e:
                recovery_context = context or {}
                recovery_context.update({
                    'operation_name': operation_name,
                    'args': args,
                    'kwargs': kwargs
                })
                
                result = await recovery_manager.attempt_recovery(
                    func, e, recovery_context, *args, **kwargs
                )
                
                if result is not None:
                    return result
                else:
                    # Re-raise original error if recovery failed
                    raise e
        
        # For non-async functions, create a sync wrapper
        def sync_wrapper(*args, **kwargs) -> T:
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator