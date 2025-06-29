import { describe, expect, it, vi, beforeEach } from 'vitest';
import { LLMResponseCache } from '../cache.js';

// Mock the logMessage function
vi.mock('../../utils/logs.js', () => ({
  logMessage: vi.fn()
}));

describe('LLMResponseCache', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it('should cache and retrieve values correctly', () => {
    const cache = new LLMResponseCache<string>();
    const messages = [{ role: 'user', content: 'test message' }];
    const temperature = 0;
    
    // Set a value in the cache
    cache.set(messages, temperature, 'test response');
    
    // Retrieve the value
    const retrieved = cache.get(messages, temperature);
    
    expect(retrieved).toBe('test response');
  });

  it('should handle cache misses', () => {
    const cache = new LLMResponseCache<string>();
    const messages = [{ role: 'user', content: 'test message' }];
    const temperature = 0;
    
    // Try to get a value that doesn't exist
    const retrieved = cache.get(messages, temperature);
    
    expect(retrieved).toBeUndefined();
  });

  it('should handle cache expiration', () => {
    const cache = new LLMResponseCache<string>({ maxAge: 1000 }); // 1 second
    const messages = [{ role: 'user', content: 'test message' }];
    const temperature = 0;
    
    // Set a value
    cache.set(messages, temperature, 'test response');
    
    // Check it exists
    expect(cache.get(messages, temperature)).toBe('test response');
    
    // Advance time by 1001ms
    vi.advanceTimersByTime(1001);
    
    // Value should be expired now
    expect(cache.get(messages, temperature)).toBeUndefined();
  });

  it('should respect cache size limits', () => {
    const cache = new LLMResponseCache<string>({ maxSize: 2 });
    
    const messages1 = [{ role: 'user', content: 'message 1' }];
    const messages2 = [{ role: 'user', content: 'message 2' }];
    const messages3 = [{ role: 'user', content: 'message 3' }];
    
    // Set 3 values (with different timestamps)
    cache.set(messages1, 0, 'response 1');
    vi.advanceTimersByTime(100);
    cache.set(messages2, 0, 'response 2');
    vi.advanceTimersByTime(100);
    cache.set(messages3, 0, 'response 3');
    
    // The oldest entry (messages1) should be evicted
    expect(cache.get(messages1, 0)).toBeUndefined();
    expect(cache.get(messages2, 0)).toBe('response 2');
    expect(cache.get(messages3, 0)).toBe('response 3');
  });

  it('should cache with different keys for different schemas', () => {
    const cache = new LLMResponseCache<string>();
    const messages = [{ role: 'user', content: 'test message' }];
    const temperature = 0;
    const schema1 = { type: 'object', properties: { name: { type: 'string' } } };
    const schema2 = { type: 'object', properties: { age: { type: 'number' } } };
    
    cache.set(messages, temperature, 'response 1', schema1);
    cache.set(messages, temperature, 'response 2', schema2);
    
    expect(cache.get(messages, temperature, schema1)).toBe('response 1');
    expect(cache.get(messages, temperature, schema2)).toBe('response 2');
  });

  it('should clear the cache when requested', () => {
    const cache = new LLMResponseCache<string>();
    const messages = [{ role: 'user', content: 'test message' }];
    
    cache.set(messages, 0, 'test response');
    expect(cache.get(messages, 0)).toBe('test response');
    
    cache.clear();
    expect(cache.get(messages, 0)).toBeUndefined();
  });

  it('should respect the enabled flag', () => {
    const cache = new LLMResponseCache<string>({ enabled: false });
    const messages = [{ role: 'user', content: 'test message' }];
    
    cache.set(messages, 0, 'test response');
    expect(cache.get(messages, 0)).toBeUndefined();
  });

  it('should handle complex message structures', () => {
    const cache = new LLMResponseCache<string>();
    const messages = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'What is the capital of France?' },
      { role: 'assistant', content: 'The capital of France is Paris.' },
      { role: 'user', content: 'And what is the capital of Germany?' }
    ];
    
    cache.set(messages, 0, 'The capital of Germany is Berlin.');
    expect(cache.get(messages, 0)).toBe('The capital of Germany is Berlin.');
  });

  it('should normalize schema objects for consistent hashing', () => {
    const cache = new LLMResponseCache<string>();
    const messages = [{ role: 'user', content: 'test message' }];
    
    // These schemas have the same properties but in different orders
    const schema1 = { 
      properties: { name: { type: 'string' }, age: { type: 'number' } },
      type: 'object'
    };
    const schema2 = {
      type: 'object',
      properties: { age: { type: 'number' }, name: { type: 'string' } }
    };
    
    cache.set(messages, 0, 'test response', schema1);
    
    // Should find the entry using schema2 even though the order is different
    expect(cache.get(messages, 0, schema2)).toBe('test response');
  });
});