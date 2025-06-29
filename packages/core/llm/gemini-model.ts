import { GoogleGenerativeAI } from "@google/generative-ai";
import OpenAI from "openai";
import { ChatCompletionMessageParam } from "openai/resources/index.mjs";
import { logMessage } from "../utils/logs.js";
import { LLMResponseCache } from "./cache.js";
import { LLM, LLMObjectResponse, LLMResponse } from "./llm.js";

export class GeminiModel implements LLM {
    public contextLength: number = 1000000;
    private genAI: GoogleGenerativeAI;
    private textCache: LLMResponseCache<LLMResponse>;
    private objectCache: LLMResponseCache<LLMObjectResponse>;
    private cacheEnabled: boolean;
    
    constructor() {
        this.genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
        
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
        
        const { geminiHistory, systemInstruction, userPrompt } = this.convertToGeminiHistory(messages);
        const modelName = process.env.GEMINI_MODEL || "gemini-2.5-flash-preview-04-17";
        
        // Log API call intent
        const messageTokenEstimate = JSON.stringify(messages).length / 4; // Rough token estimate
        logMessage('debug', `Gemini API call: ${modelName}, ~${messageTokenEstimate} tokens, temp=${temperature}`);
        
        try {
            const startTime = Date.now();
            const model = this.genAI.getGenerativeModel({
                model: modelName,
                systemInstruction: systemInstruction + "\n\n" + "The current date and time is " + new Date().toISOString(),
            });

            const chatSession = model.startChat({
                generationConfig: {
                    temperature: temperature,
                    topP: 0.95,
                    topK: 64,
                    maxOutputTokens: 65536,
                    responseMimeType: "text/plain"
                } as any,
                history: geminiHistory
            });
            
            const result = await chatSession.sendMessage(userPrompt);
            const duration = Date.now() - startTime;
            
            let responseText = result.response.text();
            
            // Log completion information
            logMessage('debug', `Gemini response received in ${duration}ms`);
            
            // Create a new copy of messages to avoid modifying the input array
            const updatedMessages = [...messages, {
                role: "assistant",
                content: responseText
            }];
            
            const response = {
                response: responseText,
                messages: updatedMessages
            };
            
            // Cache the result for temperature=0 (deterministic) responses
            if (temperature === 0) {
                this.textCache.set(messages, temperature, response);
            }
            
            return response;
        } catch (error) {
            logMessage('error', `Gemini API error: ${error.message}`);
            throw error;
        }
    }
    
    async generateObject(messages: ChatCompletionMessageParam[], schema: any, temperature: number = 0): Promise<LLMObjectResponse> {
        // Try to get from cache first (only for deterministic responses with temperature=0)
        if (temperature === 0) {
            const cached = this.objectCache.get(messages, temperature, schema);
            if (cached) {
                return cached;
            }
        }

        // Remove additionalProperties and make all properties required
        const cleanSchema = schema ? this.cleanSchemaForGemini(schema) : undefined;
        const { geminiHistory, systemInstruction, userPrompt } = this.convertToGeminiHistory(messages);
        const modelName = process.env.GEMINI_MODEL || "gemini-2.5-flash-preview-04-17";
        
        // Log API call intent
        const messageTokenEstimate = JSON.stringify(messages).length / 4; // Rough token estimate
        logMessage('debug', `Gemini JSON API call: ${modelName}, ~${messageTokenEstimate} tokens, temp=${temperature}`);
        
        try {
            const startTime = Date.now();
            const model = this.genAI.getGenerativeModel({
                model: modelName,
                systemInstruction: systemInstruction + "\n\n" + "The current date and time is " + new Date().toISOString(),
            });
            
            const chatSession = model.startChat({
                generationConfig: {
                    temperature: temperature,
                    topP: 0.95,
                    topK: 64,
                    maxOutputTokens: 65536,
                    responseMimeType: "application/json",
                    responseSchema: cleanSchema,
                },
                history: geminiHistory
            });
            
            const result = await chatSession.sendMessage(userPrompt);
            const duration = Date.now() - startTime;
            
            let responseText = result.response.text();
            
            // Log completion information
            logMessage('debug', `Gemini JSON response received in ${duration}ms`);
            
            let generatedObject;
            try {
                // Clean up any potential prefixes/suffixes while preserving arrays
                responseText = responseText.replace(/^[^[{]*/, '').replace(/[^}\]]*$/, '');
                generatedObject = JSON.parse(responseText);
            } catch (parseError) {
                logMessage('error', `JSON parsing error: ${parseError.message}`);
                throw new Error(`Failed to parse JSON response: ${parseError.message}`);
            }

            // Create a new copy of messages to avoid modifying the input array
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
            logMessage('error', `Gemini API error (JSON mode): ${error.message}`);
            throw error;
        }
    }

    private cleanSchemaForGemini(schema: any): any {
        // Create a deep copy to avoid modifying the original schema
        schema = JSON.parse(JSON.stringify(schema));
        
        // Remove $schema property
        if (schema.$schema !== undefined) {
            delete schema.$schema;
        }

        // Remove additionalProperties and optional flags
        if (schema.additionalProperties !== undefined) {
            delete schema.additionalProperties;
        }
        if (schema.optional !== undefined) {
            delete schema.optional;
        }

        // Make all properties required
        if (schema.properties && typeof schema.properties === 'object') {
            // Add a 'required' array with all property names
            schema.required = Object.keys(schema.properties);

            for (const prop of Object.values(schema.properties)) {
                if (typeof prop === 'object') {
                    this.cleanSchemaForGemini(prop); // Recurse for nested properties
                }
            }
        }

        // Handle arrays
        if (schema.items) {
            if (typeof schema.items === 'object') {
                this.cleanSchemaForGemini(schema.items); // Recurse for items in arrays
            }
        }
        return schema;
    }

    private convertToGeminiHistory(messages: OpenAI.Chat.ChatCompletionMessageParam[]): { geminiHistory: any; systemInstruction: any; userPrompt: any; } {
        const geminiHistory: any[] = [];
        let userPrompt: any;
        let systemInstruction: any;
        
        for (var i = 0; i < messages.length; i++) {
            if (messages[i].role == "system") {
                systemInstruction = messages[i].content;
                continue;
            }
            if (i == messages.length - 1) {
                userPrompt = messages[i].content;
                continue;
            }

            geminiHistory.push({
                role: messages[i].role == "assistant" ? "model" : messages[i].role,
                parts: [{ text: messages[i].content }]
            });
        }
        return { geminiHistory, systemInstruction, userPrompt };
    }
}

