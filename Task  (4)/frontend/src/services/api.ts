import { useState, useCallback } from 'react'
import { useApiKeyStore } from '@/store/apiKeyStore'
import { NegotiationRequest, NegotiationResponse, ComparisonResponse } from '@/types/api'

interface ApiClientConfig {
  baseURL: string
  timeout?: number
}

class ApiClient {
  private baseURL: string
  private timeout: number
  
  constructor(config: ApiClientConfig) {
    this.baseURL = config.baseURL || import.meta.env.VITE_API_URL
    this.timeout = config.timeout || 180000 // Increased to 180 seconds (3 minutes) for AI processing
  }
  
  private getHeaders(): HeadersInit {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    }
    
    // Get API keys from store
    const store = useApiKeyStore.getState()
    const openaiKey = store.getDecryptedKey('openai')
    const cohereKey = store.getDecryptedKey('cohere')
    const exaKey = store.getDecryptedKey('exa')
    
    // Add API keys to headers if available
    if (openaiKey) headers['X-OpenAI-API-Key'] = openaiKey
    if (cohereKey) headers['X-Cohere-API-Key'] = cohereKey
    if (exaKey) headers['X-Exa-API-Key'] = exaKey
    
    return headers
  }
  
  async negotiate(request: NegotiationRequest, mode: 'naive' | 'advanced' = 'advanced'): Promise<NegotiationResponse> {
    // Transform frontend request to match API expectations
    const apiRequest = {
      negotiation_text: request.scenario,
      user_context: {
        budget: request.budget,
        timeline: request.timeline,
        requirements: request.requirements,
        client_type: request.client_type,
        project_type: request.project_type
      }
    }
    
    let lastError: Error | null = null
    const maxRetries = 2
    
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        const response = await fetch(`${this.baseURL}/negotiate?mode=${mode}`, {
          method: 'POST',
          headers: this.getHeaders(),
          body: JSON.stringify(apiRequest),
          signal: AbortSignal.timeout(this.timeout)
        })
        
        if (!response.ok) {
          if (response.status === 401) {
            throw new Error('Invalid API key. Please check your settings.')
          }
          throw new Error(`Request failed: ${response.statusText}`)
        }
        
        return response.json()
      } catch (error: any) {
        lastError = error
        
        // Don't retry for non-timeout errors
        if (!error.message?.includes('timeout') && !error.name?.includes('Timeout')) {
          throw error
        }
        
        // Only retry if we have more attempts
        if (attempt < maxRetries - 1) {
          console.log(`Request timed out, retrying... (attempt ${attempt + 2}/${maxRetries})`)
          // Wait a bit before retrying
          await new Promise(resolve => setTimeout(resolve, 1000))
        }
      }
    }
    
    throw lastError || new Error('Request failed after retries')
  }
  
  async compareRAG(request: NegotiationRequest): Promise<ComparisonResponse> {
    // Transform frontend request to match API expectations
    const apiRequest = {
      negotiation_text: request.scenario,
      user_context: {
        budget: request.budget,
        timeline: request.timeline,
        requirements: request.requirements,
        client_type: request.client_type,
        project_type: request.project_type
      }
    }
    
    const response = await fetch(`${this.baseURL}/negotiate/compare`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(apiRequest),
      signal: AbortSignal.timeout(this.timeout * 2) // 360 seconds (6 minutes) for comparison
    })
    
    if (!response.ok) {
      if (response.status === 401) {
        throw new Error('Invalid API key. Please check your settings.')
      }
      throw new Error(`Comparison failed: ${response.statusText}`)
    }
    
    return response.json()
  }
  
  async testApiKey(_keyType: string, apiKey: string): Promise<boolean> {
    const response = await fetch(`${this.baseURL}/negotiate/test-api-key`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-OpenAI-API-Key': apiKey
      }
    })
    
    const result = await response.json()
    return result.valid
  }
}

export const apiClient = new ApiClient({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api'
})

// Hook for using API client with loading states
export const useApiClient = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  
  const negotiate = useCallback(async (request: NegotiationRequest, mode: 'naive' | 'advanced' = 'advanced') => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await apiClient.negotiate(request, mode)
      return response
    } catch (err) {
      setError(err as Error)
      throw err
    } finally {
      setLoading(false)
    }
  }, [])
  
  return { negotiate, loading, error }
}

// Hook for RAG comparison
export const useRAGComparison = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [comparisonData, setComparisonData] = useState<ComparisonResponse | null>(null)
  
  const compareRAG = useCallback(async (request: NegotiationRequest) => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await apiClient.compareRAG(request)
      setComparisonData(response)
      return response
    } catch (err) {
      setError(err as Error)
      throw err
    } finally {
      setLoading(false)
    }
  }, [])
  
  const reset = useCallback(() => {
    setComparisonData(null)
    setError(null)
  }, [])
  
  return { 
    compareRAG, 
    comparisonData, 
    loading, 
    error,
    reset,
    naiveResult: comparisonData?.naive_result,
    advancedResult: comparisonData?.advanced_result,
    metrics: comparisonData?.comparison_metrics,
    recommendation: comparisonData?.recommendation
  }
}