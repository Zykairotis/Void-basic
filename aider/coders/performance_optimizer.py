"""
Performance optimization module for Aider coders system.

This module provides comprehensive performance monitoring, optimization strategies,
and caching mechanisms using modern Python patterns and best practices.
"""

import asyncio
import functools
import time
import threading
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeAlias, Literal, Generic, TypeVar, Protocol, override
from pathlib import Path
import logging
import psutil
import os
from abc import ABC, abstractmethod

# Modern type aliases for better readability
CacheKey: TypeAlias = str
CacheValue: TypeAlias = Any
MetricValue: TypeAlias = float
OptimizationLevel: TypeAlias = Literal["none", "basic", "aggressive", "maximum"]
ResourceType: TypeAlias = Literal["memory", "cpu", "disk", "network"]

T = TypeVar('T')
P = TypeVar('P')

# =============================================================================
# Performance Monitoring and Metrics
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics with modern typing."""
    operation_name: str
    start_time: float
    end_time: float | None = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_io_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and initialize metrics."""
        if self.start_time < 0:
            raise ValueError("start_time cannot be negative")
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("end_time cannot be before start_time")

    @property
    def duration_ms(self) -> float:
        """Get operation duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    @property
    def cache_hit_ratio(self) -> float:
        """Get cache hit ratio as a percentage."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return (self.cache_hits / total_requests) * 100

    def finish(self) -> None:
        """Mark the operation as finished."""
        if self.end_time is None:
            self.end_time = time.perf_counter()

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "operation": self.operation_name,
            "duration_ms": self.duration_ms,
            "memory_mb": self.memory_usage_mb,
            "cpu_percent": self.cpu_usage_percent,
            "cache_hit_ratio": self.cache_hit_ratio,
            "errors": self.error_count,
            **self.metadata
        }


class MetricsCollector:
    """Thread-safe metrics collector with modern patterns."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics_history: deque[PerformanceMetrics] = deque(maxlen=max_history)
        self._active_metrics: dict[str, PerformanceMetrics] = {}
        self._lock = threading.RLock()
        self._aggregated_stats: dict[str, dict[str, float]] = defaultdict(dict)
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def start_operation(self, operation_name: str, **metadata: Any) -> str:
        """Start tracking a new operation."""
        operation_id = f"{operation_name}_{int(time.perf_counter() * 1000000)}"

        with self._lock:
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=time.perf_counter(),
                metadata=metadata
            )
            self._active_metrics[operation_id] = metrics

        return operation_id

    def finish_operation(self, operation_id: str, **additional_metadata: Any) -> PerformanceMetrics:
        """Finish tracking an operation and collect final metrics."""
        with self._lock:
            if operation_id not in self._active_metrics:
                raise ValueError(f"Operation {operation_id} not found or already finished")

            metrics = self._active_metrics.pop(operation_id)
            metrics.finish()

            # Add additional metadata
            metrics.metadata.update(additional_metadata)

            # Collect system metrics
            self._collect_system_metrics(metrics)

            # Store in history
            self._metrics_history.append(metrics)

            # Update aggregated statistics
            self._update_aggregated_stats(metrics)

        return metrics

    def _collect_system_metrics(self, metrics: PerformanceMetrics) -> None:
        """Collect system performance metrics."""
        try:
            process = psutil.Process(os.getpid())

            # Memory usage
            memory_info = process.memory_info()
            metrics.memory_usage_mb = memory_info.rss / (1024 * 1024)

            # CPU usage (average over a short period)
            metrics.cpu_usage_percent = process.cpu_percent()

            # Disk I/O
            io_counters = process.io_counters()
            metrics.disk_io_bytes = io_counters.read_bytes + io_counters.write_bytes

        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            # System metrics not available on this platform
            pass

    def _update_aggregated_stats(self, metrics: PerformanceMetrics) -> None:
        """Update aggregated statistics for the operation."""
        op_name = metrics.operation_name
        duration = metrics.duration_ms

        if op_name not in self._aggregated_stats:
            self._aggregated_stats[op_name] = {
                "count": 0,
                "total_duration": 0.0,
                "min_duration": float('inf'),
                "max_duration": 0.0,
                "avg_duration": 0.0
            }

        stats = self._aggregated_stats[op_name]
        stats["count"] += 1
        stats["total_duration"] += duration
        stats["min_duration"] = min(stats["min_duration"], duration)
        stats["max_duration"] = max(stats["max_duration"], duration)
        stats["avg_duration"] = stats["total_duration"] / stats["count"]

    def get_operation_stats(self, operation_name: str) -> dict[str, Any]:
        """Get aggregated statistics for a specific operation."""
        with self._lock:
            return self._aggregated_stats.get(operation_name, {}).copy()

    def get_recent_metrics(self, limit: int = 100) -> List[PerformanceMetrics]:
        """Get recent performance metrics."""
        with self._lock:
            return list(self._metrics_history)[-limit:]

    def generate_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        with self._lock:
            report = {
                "timestamp": time.time(),
                "total_operations": len(self._metrics_history),
                "active_operations": len(self._active_metrics),
                "operation_stats": dict(self._aggregated_stats),
                "system_info": self._get_system_info()
            }

            # Add trend analysis
            if len(self._metrics_history) >= 10:
                report["trends"] = self._analyze_trends()

            return report

    def _get_system_info(self) -> dict[str, Any]:
        """Get current system information."""
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
        except Exception:
            return {"error": "System info not available"}

    def _analyze_trends(self) -> dict[str, Any]:
        """Analyze performance trends from recent metrics."""
        recent_metrics = list(self._metrics_history)[-50:]  # Last 50 operations

        if not recent_metrics:
            return {}

        # Calculate trends
        durations = [m.duration_ms for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics if m.memory_usage_mb > 0]

        trends = {}

        if durations:
            trends["avg_duration_trend"] = sum(durations) / len(durations)
            trends["duration_std_dev"] = (
                sum((d - trends["avg_duration_trend"]) ** 2 for d in durations) / len(durations)
            ) ** 0.5

        if memory_usage:
            trends["avg_memory_trend"] = sum(memory_usage) / len(memory_usage)

        return trends


# =============================================================================
# Advanced Caching System
# =============================================================================

class CacheProtocol(Protocol):
    """Protocol for cache implementations."""

    def get(self, key: CacheKey) -> CacheValue | None: ...
    def set(self, key: CacheKey, value: CacheValue, ttl: int | None = None) -> None: ...
    def delete(self, key: CacheKey) -> bool: ...
    def clear(self) -> None: ...
    def size(self) -> int: ...


@dataclass
class CacheEntry:
    """Cache entry with metadata and TTL support."""
    value: CacheValue
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: int | None = None

    def __post_init__(self):
        """Initialize cache entry."""
        if self.created_at <= 0:
            self.created_at = time.time()
        if self.last_accessed <= 0:
            self.last_accessed = self.created_at

    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at

    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUCache(Generic[T]):
    """Thread-safe LRU cache with TTL support and modern patterns."""

    def __init__(self, max_size: int = 256, default_ttl: int | None = None):
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[CacheKey, CacheEntry] = {}
        self._access_order: deque[CacheKey] = deque()
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get(self, key: CacheKey) -> T | None:
        """Get value from cache with LRU tracking."""
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            entry = self._cache[key]

            # Check TTL expiration
            if entry.is_expired:
                self.delete(key)
                self._stats["misses"] += 1
                return None

            # Update access information
            entry.touch()

            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)

            self._stats["hits"] += 1
            return entry.value

    def set(self, key: CacheKey, value: T, ttl: int | None = None) -> None:
        """Set value in cache with optional TTL."""
        with self._lock:
            ttl = ttl or self.default_ttl

            if key in self._cache:
                # Update existing entry
                self._cache[key].value = value
                self._cache[key].created_at = time.time()
                self._cache[key].ttl_seconds = ttl
                self._cache[key].touch()

                # Move to end
                self._access_order.remove(key)
                self._access_order.append(key)
            else:
                # New entry
                if len(self._cache) >= self.max_size:
                    self._evict_lru()

                entry = CacheEntry(
                    value=value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    ttl_seconds=ttl
                )

                self._cache[key] = entry
                self._access_order.append(key)

    def delete(self, key: CacheKey) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.remove(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_order:
            return

        lru_key = self._access_order.popleft()
        if lru_key in self._cache:
            del self._cache[lru_key]
            self._stats["evictions"] += 1

    @property
    def hit_ratio(self) -> float:
        """Get cache hit ratio."""
        total = self._stats["hits"] + self._stats["misses"]
        if total == 0:
            return 0.0
        return self._stats["hits"] / total

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_ratio": self.hit_ratio,
                **self._stats.copy()
            }


# =============================================================================
# Performance Optimization Strategies
# =============================================================================

class OptimizationStrategy(ABC):
    """Abstract base for optimization strategies."""

    @abstractmethod
    def can_optimize(self, context: dict[str, Any]) -> bool:
        """Check if this strategy can be applied to the given context."""
        pass

    @abstractmethod
    def apply_optimization(self, target: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Apply optimization to target object."""
        pass

    @property
    @abstractmethod
    def optimization_type(self) -> ResourceType:
        """Get the type of resource this strategy optimizes."""
        pass


class MemoryOptimizationStrategy(OptimizationStrategy):
    """Strategy for optimizing memory usage."""

    @override
    def can_optimize(self, context: dict[str, Any]) -> bool:
        """Check if memory optimization is beneficial."""
        memory_usage = context.get("memory_usage_mb", 0)
        return memory_usage > 100  # Optimize if using more than 100MB

    @override
    def apply_optimization(self, target: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Apply memory optimizations."""
        optimizations_applied = []

        # Enable object pooling if available
        if hasattr(target, 'enable_object_pooling'):
            target.enable_object_pooling()
            optimizations_applied.append("object_pooling")

        # Clear unnecessary caches
        if hasattr(target, 'clear_temporary_cache'):
            target.clear_temporary_cache()
            optimizations_applied.append("cache_cleanup")

        # Use weak references where possible
        if hasattr(target, 'use_weak_references'):
            target.use_weak_references(True)
            optimizations_applied.append("weak_references")

        return {
            "strategy": "memory_optimization",
            "optimizations": optimizations_applied,
            "estimated_savings_mb": len(optimizations_applied) * 10
        }

    @property
    @override
    def optimization_type(self) -> ResourceType:
        return "memory"


class CpuOptimizationStrategy(OptimizationStrategy):
    """Strategy for optimizing CPU usage."""

    @override
    def can_optimize(self, context: dict[str, Any]) -> bool:
        """Check if CPU optimization is beneficial."""
        cpu_usage = context.get("cpu_usage_percent", 0)
        return cpu_usage > 70  # Optimize if using more than 70% CPU

    @override
    def apply_optimization(self, target: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Apply CPU optimizations."""
        optimizations_applied = []

        # Enable caching of expensive operations
        if hasattr(target, 'enable_result_caching'):
            target.enable_result_caching()
            optimizations_applied.append("result_caching")

        # Use async operations where possible
        if hasattr(target, 'enable_async_processing'):
            target.enable_async_processing()
            optimizations_applied.append("async_processing")

        # Optimize algorithms
        if hasattr(target, 'use_optimized_algorithms'):
            target.use_optimized_algorithms()
            optimizations_applied.append("algorithm_optimization")

        return {
            "strategy": "cpu_optimization",
            "optimizations": optimizations_applied,
            "estimated_cpu_reduction_percent": len(optimizations_applied) * 15
        }

    @property
    @override
    def optimization_type(self) -> ResourceType:
        return "cpu"


class PerformanceOptimizer:
    """Comprehensive performance optimizer with multiple strategies."""

    def __init__(self, metrics_collector: MetricsCollector | None = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.strategies: list[OptimizationStrategy] = [
            MemoryOptimizationStrategy(),
            CpuOptimizationStrategy()
        ]
        self._optimization_cache = LRUCache[dict[str, Any]](max_size=100)
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def analyze_performance(self, target: Any, operation_name: str) -> dict[str, Any]:
        """Analyze performance characteristics of a target object."""
        analysis_key = f"{type(target).__name__}_{operation_name}"

        # Check cache first
        cached_analysis = self._optimization_cache.get(analysis_key)
        if cached_analysis is not None:
            return cached_analysis

        # Perform fresh analysis
        operation_id = self.metrics_collector.start_operation(
            f"analyze_{operation_name}",
            target_type=type(target).__name__
        )

        try:
            context = self._gather_performance_context(target)

            # Identify applicable strategies
            applicable_strategies = [
                strategy for strategy in self.strategies
                if strategy.can_optimize(context)
            ]

            analysis = {
                "context": context,
                "applicable_strategies": [
                    strategy.optimization_type for strategy in applicable_strategies
                ],
                "optimization_potential": len(applicable_strategies) > 0,
                "timestamp": time.time()
            }

            # Cache the analysis
            self._optimization_cache.set(analysis_key, analysis, ttl=300)  # 5 minutes

            return analysis

        finally:
            self.metrics_collector.finish_operation(operation_id)

    def optimize(self, target: Any, optimization_level: OptimizationLevel = "basic") -> dict[str, Any]:
        """Apply optimizations to target object."""
        operation_id = self.metrics_collector.start_operation(
            "optimize",
            target_type=type(target).__name__,
            level=optimization_level
        )

        try:
            context = self._gather_performance_context(target)
            results = {
                "optimizations_applied": [],
                "estimated_improvements": {},
                "level": optimization_level
            }

            # Apply strategies based on optimization level
            strategies_to_apply = self._select_strategies(optimization_level, context)

            for strategy in strategies_to_apply:
                if strategy.can_optimize(context):
                    try:
                        optimization_result = strategy.apply_optimization(target, context)
                        results["optimizations_applied"].append(optimization_result)

                        # Merge estimated improvements
                        for key, value in optimization_result.items():
                            if key.startswith("estimated_"):
                                results["estimated_improvements"][key] = value

                    except Exception as e:
                        self.logger.warning(f"Optimization strategy {strategy.__class__.__name__} failed: {e}")

            return results

        finally:
            self.metrics_collector.finish_operation(operation_id)

    def _gather_performance_context(self, target: Any) -> dict[str, Any]:
        """Gather performance context about the target object."""
        context = {
            "target_type": type(target).__name__,
            "timestamp": time.time()
        }

        # Get system metrics
        try:
            process = psutil.Process(os.getpid())
            context.update({
                "memory_usage_mb": process.memory_info().rss / (1024 * 1024),
                "cpu_usage_percent": process.cpu_percent(),
                "thread_count": process.num_threads()
            })
        except Exception:
            pass

        # Get object-specific metrics
        if hasattr(target, 'get_performance_metrics'):
            context.update(target.get_performance_metrics())

        return context

    def _select_strategies(self, level: OptimizationLevel, context: dict[str, Any]) -> list[OptimizationStrategy]:
        """Select optimization strategies based on level and context."""
        if level == "none":
            return []

        strategies = []

        # Basic level: only safe, low-impact optimizations
        if level in ["basic", "aggressive", "maximum"]:
            strategies.extend([s for s in self.strategies if s.optimization_type in ["memory", "cpu"]])

        # Aggressive level: more intensive optimizations
        if level in ["aggressive", "maximum"]:
            # Add more aggressive strategies when implemented
            pass

        # Maximum level: all available optimizations
        if level == "maximum":
            strategies = self.strategies.copy()

        return strategies


# =============================================================================
# Context Managers and Decorators
# =============================================================================

@contextmanager
def performance_monitoring(operation_name: str, metrics_collector: MetricsCollector | None = None):
    """Context manager for automatic performance monitoring."""
    collector = metrics_collector or MetricsCollector()
    operation_id = collector.start_operation(operation_name)

    try:
        yield collector
    finally:
        metrics = collector.finish_operation(operation_id)
        logging.getLogger("performance").info(
            f"Operation '{operation_name}' completed in {metrics.duration_ms:.2f}ms"
        )


@asynccontextmanager
async def async_performance_monitoring(operation_name: str, metrics_collector: MetricsCollector | None = None):
    """Async context manager for performance monitoring."""
    collector = metrics_collector or MetricsCollector()
    operation_id = collector.start_operation(operation_name)

    try:
        yield collector
    finally:
        metrics = collector.finish_operation(operation_id)
        logging.getLogger("performance").info(
            f"Async operation '{operation_name}' completed in {metrics.duration_ms:.2f}ms"
        )


def performance_monitor(operation_name: str | None = None, metrics_collector: MetricsCollector | None = None):
    """Decorator for automatic performance monitoring of functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = operation_name or f"{func.__module__}.{func.__name__}"
        collector = metrics_collector or MetricsCollector()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with performance_monitoring(name, collector):
                return func(*args, **kwargs)

        return wrapper
    return decorator


def async_performance_monitor(operation_name: str | None = None, metrics_collector: MetricsCollector | None = None):
    """Decorator for automatic performance monitoring of async functions."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        name = operation_name or f"{func.__module__}.{func.__name__}"
        collector = metrics_collector or MetricsCollector()

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with async_performance_monitoring(name, collector):
                return await func(*args, **kwargs)

        return wrapper
    return decorator


def cached_result(cache: LRUCache[Any] | None = None, ttl: int | None = None, key_func: Callable[..., str] | None = None):
    """Decorator for caching function results with configurable cache and TTL."""
    result_cache = cache or LRUCache[Any](max_size=256, default_ttl=ttl)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = "|".join(key_parts)

            # Try to get from cache
            cached_result = result_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Compute and cache result
            result = func(*args, **kwargs)
            result_cache.set(cache_key, result, ttl=ttl)
            return result

        # Add cache management methods to the wrapper
        wrapper.cache = result_cache  # type: ignore
        wrapper.cache_clear = result_cache.clear  # type: ignore
        wrapper.cache_info = result_cache.get_stats  # type: ignore

        return wrapper
    return decorator


# =============================================================================
# Resource Management and Pooling
# =============================================================================

class ResourcePool(Generic[T]):
    """Generic resource pool with lifecycle management."""

    def __init__(self, factory: Callable[[], T], max_size: int = 10, min_size: int = 1):
        if max_size <= 0 or min_size < 0 or min_size > max_size:
            raise ValueError("Invalid pool size parameters")

        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self._pool: deque[T] = deque()
        self._active_resources: weakref.WeakSet[T] = weakref.WeakSet()
        self._lock = threading.RLock()
        self._created_count = 0

        # Initialize minimum resources
        for _ in range(min_size):
            resource = self.factory()
            self._pool.append(resource)
            self._created_count += 1

    def acquire(self) -> T:
        """Acquire a resource from the pool."""
        with self._lock:
            if self._pool:
                resource = self._pool.popleft()
            elif self._created_count < self.max_size:
                resource = self.factory()
                self._created_count += 1
            else:
                raise RuntimeError("Resource pool exhausted")

            self._active_resources.add(resource)
            return resource

    def release(self, resource: T) -> None:
        """Release a resource back to the pool."""
        with self._lock:
            if resource in self._active_resources:
                self._active_resources.discard(resource)

                # Return to pool if under max size
                if len(self._pool) < self.max_size:
                    self._pool.append(resource)
                else:
                    # Pool is full, let resource be garbage collected
                    self._created_count -= 1

    @contextmanager
    def resource(self):
        """Context manager for automatic resource lifecycle management."""
        resource = self.acquire()
        try:
            yield resource
        finally:
            self.release(resource)

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "pool_size": len(self._pool),
                "active_resources": len(self._active_resources),
                "created_count": self._created_count,
                "max_size": self.max_size,
                "utilization": len(self._active_resources) / self.max_size
            }


# =============================================================================
# Global Performance Manager
# =============================================================================

class GlobalPerformanceManager:
    """Singleton performance manager for the entire application."""

    _instance: 'GlobalPerformanceManager | None' = None
    _lock = threading.Lock()

    def __new__(cls) -> 'GlobalPerformanceManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.metrics_collector = MetricsCollector(max_history=5000)
        self.optimizer = PerformanceOptimizer(self.metrics_collector)
        self.global_cache = LRUCache[Any](max_size=1000, default_ttl=3600)  # 1 hour TTL
        self._resource_pools: dict[str, ResourcePool[Any]] = {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def get_or_create_resource_pool(self, name: str, factory: Callable[[], Any], max_size: int = 10) -> ResourcePool[Any]:
        """Get or create a named resource pool."""
        if name not in self._resource_pools:
            self._resource_pools[name] = ResourcePool[Any](factory, max_size)
        return self._resource_pools[name]

    def get_resource_pool(self, name: str) -> ResourcePool[Any] | None:
        """Get an existing resource pool by name."""
        return self._resource_pools.get(name)

    def monitor_operation(self, operation_name: str):
        """Get a context manager for monitoring an operation."""
        return performance_monitoring(operation_name, self.metrics_collector)

    def optimize_target(self, target: Any, level: OptimizationLevel = "basic") -> dict[str, Any]:
        """Optimize a target object using the global optimizer."""
        return self.optimizer.optimize(target, level)

    def get_global_stats(self) -> dict[str, Any]:
        """Get comprehensive global performance statistics."""
        stats = {
            "metrics": self.metrics_collector.generate_performance_report(),
            "cache_stats": self.global_cache.get_stats(),
            "resource_pools": {
                name: pool.get_stats()
                for name, pool in self._resource_pools.items()
            }
        }
        return stats

    def cleanup(self) -> None:
        """Cleanup global performance manager resources."""
        self.global_cache.clear()
        self._resource_pools.clear()
        self.logger.info("Global performance manager cleaned up")


# =============================================================================
# Convenience Functions
# =============================================================================

def get_global_performance_manager() -> GlobalPerformanceManager:
    """Get the global performance manager instance."""
    return GlobalPerformanceManager()


def monitor_performance(operation_name: str):
    """Convenience function to monitor performance globally."""
    manager = get_global_performance_manager()
    return manager.monitor_operation(operation_name)


def optimize_object(target: Any, level: OptimizationLevel = "basic") -> dict[str, Any]:
    """Convenience function to optimize an object globally."""
    manager = get_global_performance_manager()
    return manager.optimize_target(target, level)


def get_performance_stats() -> dict[str, Any]:
    """Convenience function to get global performance statistics."""
    manager = get_global_performance_manager()
    return manager.get_global_stats()


# =============================================================================
# Module-level exports and initialization
# =============================================================================

__all__ = [
    # Core classes
    "PerformanceMetrics",
    "MetricsCollector",
    "LRUCache",
    "PerformanceOptimizer",
    "ResourcePool",
    "GlobalPerformanceManager",

    # Strategies
    "OptimizationStrategy",
    "MemoryOptimizationStrategy",
    "CpuOptimizationStrategy",

    # Context managers and decorators
    "performance_monitoring",
    "async_performance_monitoring",
    "performance_monitor",
    "async_performance_monitor",
    "cached_result",

    # Convenience functions
    "get_global_performance_manager",
    "monitor_performance",
    "optimize_object",
    "get_performance_stats",

    # Type aliases
    "CacheKey",
    "CacheValue",
    "MetricValue",
    "OptimizationLevel",
    "ResourceType"
]
