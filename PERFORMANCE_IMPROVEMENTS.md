# SuperGlue Performance and Caching Improvements

This pull request introduces significant performance optimizations and caching mechanisms to enhance SuperGlue's efficiency, reduce API costs, and improve responsiveness.

## Key Improvements

### 1. LLM Response Caching

- **In-memory LLM Cache**: Added a lightweight caching system for LLM responses that caches deterministic (temperature=0) requests to avoid redundant API calls.
- **Configurable Cache Parameters**: Cache size and TTL configurable via environment variables.
- **Smart Hashing**: Uses message content, temperature, and schema for cache key generation.
- **Cache Management**: Automatic cache cleanup and eviction strategies to prevent memory leaks.

### 2. Redis Performance Optimizations

- **Pipeline Operations**: Replaced multiple sequential Redis operations with pipeline batching for significant performance improvement.
- **Connection Resilience**: Enhanced Redis connection with TLS support, configurable timeouts, and intelligent reconnect strategies.
- **Error Handling**: Improved error handling and logging throughout the Redis service.
- **Early Returns**: Added early returns for empty result sets to avoid unnecessary processing.

### 3. Memory Usage Improvements

- **Reduced Duplicated Data**: Fixed several instances where data was being unnecessarily duplicated.
- **Deep Cloning**: Added selective deep cloning to prevent unintended mutations of shared objects.
- **Optimized Input Handling**: Better handling of input arrays and objects to reduce memory footprint.

### 4. Comprehensive Logging

- **LLM API Call Tracking**: Added detailed logging for LLM API calls with estimated token counts and response times.
- **Performance Metrics**: Included timing information for expensive operations.
- **Error Context**: Enhanced error messages with more context for easier debugging.

## Configuration Options

New environment variables:

- `LLM_CACHE_ENABLED` - Set to 'false' to disable LLM caching (default: true)
- `LLM_CACHE_TTL` - Cache entry lifetime in milliseconds (default: 3600000 / 1 hour)
- `LLM_CACHE_SIZE` - Maximum number of entries in cache (default: 1000)

## Testing

Added comprehensive tests for the new caching system to ensure reliability and correctness.

## Implementation Notes

1. The LLM caching system automatically detects and skips caching for non-deterministic calls (temperature > 0).
2. Redis pipeline operations significantly reduce network overhead when fetching multiple items.
3. All changes maintain backward compatibility with existing code.

These improvements enhance SuperGlue's efficiency particularly for workflows with repeated similar LLM calls and for applications with high database read volumes.