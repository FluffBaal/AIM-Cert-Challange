// API Request and Response Types

export interface NegotiationRequest {
  scenario: string
  budget: number
  timeline: string
  requirements: string[]
  client_type: string
  project_type: string
}

export interface NegotiationResponse {
  success: boolean
  partial_failure?: boolean
  degraded?: boolean
  rewritten_text: string
  strategy: {
    techniques: string[]
    confidence: number
    approach: string
  }
  market_data?: {
    min_rate: number
    max_rate: number
    median_rate: number
    average_rate?: number
    currency: string
    sources: string[]
    data_points: number
    location?: string
    skill?: string
    confidence: number
    note?: string
  } | null
  insights: string[]
  confidence: number
  mode: string
  agents_used: string[]
  retrieval_time: number
  error_recovery?: string | null
  request_id: string
  timestamp: string
  retrieved_contexts: string[]
  avg_similarity_score?: number | null
  context_corridor_used: boolean
  retrieval_details?: {
    documents_searched: string
    chunks_retrieved: number
    relevance_threshold: number
    mode: string
    context_corridor_used: boolean
  }
  processing_steps?: Array<{
    id: string
    name: string
    description: string
    status: 'pending' | 'active' | 'completed' | 'skipped'
    startTime?: number
    endTime?: number
  }>
  detailed_contexts?: Array<{
    content: string
    source: string
    chapter?: string
    page?: number
    relevance_score: number
    technique?: string
    metadata?: Record<string, any>
  }>
}

export interface ComparisonResponse {
  naive_result: NegotiationResponse
  advanced_result: NegotiationResponse
  comparison_metrics: {
    response_time_naive: number
    response_time_advanced: number
    accuracy_score_naive: number
    accuracy_score_advanced: number
    relevance_score_naive: number
    relevance_score_advanced: number
  }
  recommendation: {
    preferred_approach: 'naive' | 'advanced'
    reasoning: string
    improvements: string[]
  }
}

export interface ApiTestResult {
  valid: boolean
  error?: string
}