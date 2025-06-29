import OpenAI from "openai";
import { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { logMessage } from "../utils/logs.js";
import { addNullableToOptional } from "../utils/tools.js";
import { LLMResponseCache } from "./cache.js";
import { LLM, LLMObjectResponse, LLMResponse } from "./llm.js";

export class OpenAIModel implements LLM {
  public contextLength: number = 128000;
  private model: OpenAI;
  private textCache: LLMResponseCache<LLMResponse>;
  private objectCache: LLMResponseCache<LLMObjectResponse>;
  private cacheEnabled: boolean;

  constructor() {
    this.model = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY || "",
      baseURL: process.env.OPENAI_BASE_URL,
    });
    
    // Initialize caches based on environment variables
    this.cacheEnabled = process.env.LLM_CACHE_ENABLED !== 'false';
    const cacheTTL = parseInt(process.env.LLM_CACHE_TTL || '3600000', 10);
    const cacheSize = parseInt(process.env.LLM_CACHE_SIZE || '1000', 10);
    
    this.textCache = new LLMResponseCache<LLMResponse>({
      enabled: this.cacheEnabled,
      maxAge: cacheTTL,
      maxSize: cacheSize
    });
    
    this.objectCache = new LLMResponseCache<LLMObjectResponse>({
      enabled: this.cacheEnabled,
      maxAge: cacheTTL,
      maxSize: cacheSize
    });
  }

  async generateText(messages: ChatCompletionMessageParam[], temperature: number = 0): Promise<LLMResponse> {
    // Try to get from cache first (only for deterministic responses with temperature=0)
    if (temperature === 0) {
      const cached = this.textCache.get(messages, temperature);
      if (cached) {
        return cached;
      }
    }
    
    // o models don't support temperature
    if (process.env.OPENAI_MODEL?.startsWith('o')) {
      temperature = undefined;
    }
    
    // Add timestamp message
    const dateMessage = {
      role: "system",
      content: "The current date and time is " + new Date().toISOString()
    } as ChatCompletionMessageParam;

    // Log API call intent
    const modelName = process.env.OPENAI_MODEL || "gpt-4o";
    const messageTokenEstimate = JSON.stringify(messages).length / 4; // Rough token estimate
    logMessage('debug', `OpenAI API call: ${modelName}, ~${messageTokenEstimate} tokens, temp=${temperature}`);
    
    try {
      const startTime = Date.now();
      const result = await this.model.chat.completions.create({
        messages: [dateMessage, ...messages],
        model: modelName,
        temperature: temperature
      });
      const duration = Date.now() - startTime;
      
      let responseText = result.choices[0].message.content;
      
      // Log completion information
      logMessage('debug', `OpenAI response received in ${duration}ms`);

      // Add response to messages history
      const updatedMessages = [...messages, {
        role: "assistant",
        content: responseText
      }];

      const response = {
        response: responseText,
        messages: updatedMessages
      } as LLMResponse;
      
      // Cache the result for temperature=0 (deterministic) responses
      if (temperature === 0) {
        this.textCache.set(messages, temperature, response);
      }

      return response;
    } catch (error) {
      logMessage('error', `OpenAI API error: ${error.message}`);
      throw error;
    }
  }

  private enforceStrictSchema(schema: any, isRoot: boolean) {
    if (!schema || typeof schema !== 'object') return schema;

    // wrap non-object in object with ___results key
    if (isRoot && schema.type !== 'object') {
      schema = {
        type: 'object',
        properties: {
          ___results: schema,
        },
      };
    }

    if (schema.type === 'object' || schema.type === 'array') {
      schema.additionalProperties = false;
      schema.strict = true;
      if (schema.properties) {
        // Only set required for the top-level schema
        schema.required = Object.keys(schema.properties);
        delete schema.patternProperties;
        // Recursively process nested properties
        Object.values(schema.properties).forEach(prop => this.enforceStrictSchema(prop, false));
      }
      if (schema.items) {
        schema.items = this.enforceStrictSchema(schema.items, false);
        delete schema.minItems;
        delete schema.maxItems;
      }
    }

    return schema;
  };

  async generateObject(messages: ChatCompletionMessageParam[], schema: any, temperature: number = 0): Promise<LLMObjectResponse> {
    // Try to get from cache first (only for deterministic responses with temperature=0)
    if (temperature === 0) {
      const cached = this.objectCache.get(messages, temperature, schema);
      if (cached) {
        return cached;
      }
    }
    
    // Prepare schema
    schema = addNullableToOptional(schema)
    schema = this.enforceStrictSchema(schema, true);
    
    // o models don't support temperature
    if (process.env.OPENAI_MODEL?.startsWith('o')) {
      temperature = undefined;
    }
    
    const responseFormat = schema ? 
      { type: "json_schema", json_schema: { name: "response", strict: true, schema: schema } } : 
      { type: "json_object" };
    
    const dateMessage = {
      role: "system",
      content: "The current date and time is " + new Date().toISOString()
    } as ChatCompletionMessageParam;

    // Log API call intent
    const modelName = process.env.OPENAI_MODEL || "gpt-4o";
    const messageTokenEstimate = JSON.stringify(messages).length / 4; // Rough token estimate
    logMessage('debug', `OpenAI JSON API call: ${modelName}, ~${messageTokenEstimate} tokens, temp=${temperature}`);
    
    try {
      const startTime = Date.now();
      const result = await this.model.chat.completions.create({
        messages: [dateMessage, ...messages],
        model: modelName,
        temperature: temperature,
        response_format: responseFormat as any,
      });
      const duration = Date.now() - startTime;
      
      let responseText = result.choices[0].message.content;
      
      // Log completion information
      logMessage('debug', `OpenAI JSON response received in ${duration}ms`);
      
      let generatedObject;
      try {
        generatedObject = JSON.parse(responseText);
        if (generatedObject.___results) {
          generatedObject = generatedObject.___results;
        }
      } catch (parseError) {
        logMessage('error', `JSON parsing error: ${parseError.message}`);
        throw new Error(`Failed to parse JSON response: ${parseError.message}`);
      }

      // Add response to messages history
      const updatedMessages = [...messages, {
        role: "assistant",
        content: responseText
      }];

      const response = {
        response: generatedObject,
        messages: updatedMessages
      } as LLMObjectResponse;
      
      // Cache the result for temperature=0 (deterministic) responses
      if (temperature === 0) {
        this.objectCache.set(messages, temperature, response, schema);
      }

      return response;
    } catch (error) {
      logMessage('error', `OpenAI API error (JSON mode): ${error.message}`);
      throw error;
    }
  }
}

