import { createHash } from 'crypto';
import { logMessage } from '../utils/logs.js';

interface CacheEntry<T> {
  timestamp: number;
  value: T;
}

/**
 * A lightweight in-memory cache for LLM responses
 * Helps reduce API costs and latency by caching responses
 */
export class LLMResponseCache<T> {
  private cache: Map<string, CacheEntry<T>> = new Map();
  private readonly maxAge: number;
  private readonly maxSize: number;
  private readonly enabled: boolean;
  
  /**
   * Creates a new LLM response cache
   * @param options Configuration options
   * @param options.maxAge Maximum age of entries in milliseconds (default: 1 hour)
   * @param options.maxSize Maximum number of entries to keep in cache (default: 1000)
   * @param options.enabled Whether the cache is enabled (default: true)
   */
  constructor(options?: {
    maxAge?: number;
    maxSize?: number;
    enabled?: boolean;
  }) {
    this.maxAge = options?.maxAge || 60 * 60 * 1000; // Default: 1 hour
    this.maxSize = options?.maxSize || 1000; // Default: 1000 entries
    this.enabled = options?.enabled !== false; // Default: true
  }

  /**
   * Generate a cache key from messages and other parameters
   * @param messages The messages to generate a key for
   * @param temperature The temperature parameter
   * @param schema Optional schema for JSON responses
   * @returns A deterministic hash key
   */
  private generateKey(messages: any[], temperature: number, schema?: any): string {
    // Sort schema keys to ensure consistent order
    const normalizedSchema = schema ? this.normalizeObject(schema) : undefined;
    
    // Create a composite key from messages, temperature, and schema
    const keyObj = {
      messages: messages.map(msg => ({
        role: msg.role,
        content: msg.content
      })),
      temperature,
      schema: normalizedSchema
    };
    
    // Generate a consistent hash
    return createHash('md5')
      .update(JSON.stringify(keyObj))
      .digest('hex');
  }

  /**
   * Helper to normalize objects for consistent hashing
   * @param obj The object to normalize
   * @returns A normalized object with sorted keys
   */
  private normalizeObject(obj: any): any {
    if (typeof obj !== 'object' || obj === null) {
      return obj;
    }
    
    if (Array.isArray(obj)) {
      return obj.map(item => this.normalizeObject(item));
    }
    
    return Object.keys(obj)
      .sort()
      .reduce((result, key) => {
        result[key] = this.normalizeObject(obj[key]);
        return result;
      }, {});
  }

  /**
   * Get a value from the cache
   * @param messages The message array (used for generating the key)
   * @param temperature The temperature parameter
   * @param schema Optional schema for JSON responses
   * @returns The cached value or undefined if not found or expired
   */
  get(messages: any[], temperature: number, schema?: any): T | undefined {
    if (!this.enabled) return undefined;
    
    const key = this.generateKey(messages, temperature, schema);
    const entry = this.cache.get(key);
    
    // Return undefined if entry doesn't exist or is expired
    if (!entry || Date.now() - entry.timestamp > this.maxAge) {
      if (entry) {
        // Clean up expired entry
        this.cache.delete(key);
        logMessage('debug', `LLM cache entry expired: ${key.substring(0, 8)}...`);
      }
      return undefined;
    }
    
    logMessage('info', `LLM cache hit: ${key.substring(0, 8)}...`);
    return entry.value;
  }

  /**
   * Store a value in the cache
   * @param messages The message array (used for generating the key)
   * @param temperature The temperature parameter
   * @param value The value to store
   * @param schema Optional schema for JSON responses
   */
  set(messages: any[], temperature: number, value: T, schema?: any): void {
    if (!this.enabled) return;
    
    const key = this.generateKey(messages, temperature, schema);
    
    // Enforce cache size limit
    if (this.cache.size >= this.maxSize) {
      // Get the oldest entry
      let oldest: [string, CacheEntry<T>] | undefined;
      let oldestTime = Infinity;
      
      for (const [entryKey, entry] of this.cache.entries()) {
        if (entry.timestamp < oldestTime) {
          oldestTime = entry.timestamp;
          oldest = [entryKey, entry];
        }
      }
      
      // Remove the oldest entry
      if (oldest) {
        this.cache.delete(oldest[0]);
        logMessage('debug', `LLM cache evicted oldest entry: ${oldest[0].substring(0, 8)}...`);
      }
    }
    
    // Store the new entry
    this.cache.set(key, {
      timestamp: Date.now(),
      value
    });
    
    logMessage('debug', `LLM cache set: ${key.substring(0, 8)}...`);
  }

  /**
   * Clear the entire cache
   */
  clear(): void {
    this.cache.clear();
    logMessage('info', 'LLM cache cleared');
  }

  /**
   * Get the current size of the cache
   */
  get size(): number {
    return this.cache.size;
  }
}