import React, { useState } from 'react'
import { 
  ArrowRight, 
  Zap, 
  Users, 
  TrendingUp,
  Award,
  RefreshCw,
  AlertCircle
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ResponseDisplay } from './ResponseDisplay'
import { MessageInput } from './MessageInput'
import { useRAGComparison } from '@/services/api'
import { useApiKeyStore } from '@/store/apiKeyStore'
import { NegotiationRequest } from '@/types/api'

interface MetricCardProps {
  title: string
  naiveValue: number
  advancedValue: number
  unit?: string
  higherIsBetter?: boolean
}

const MetricCard: React.FC<MetricCardProps> = ({ 
  title, 
  naiveValue, 
  advancedValue, 
  unit = '', 
  higherIsBetter = true 
}) => {
  const isBetter = higherIsBetter 
    ? advancedValue > naiveValue 
    : advancedValue < naiveValue
  
  const improvementPercent = naiveValue > 0 
    ? Math.abs(((advancedValue - naiveValue) / naiveValue) * 100)
    : 0

  return (
    <div className="p-4 bg-card rounded-lg border">
      <h4 className="font-medium text-sm text-muted-foreground mb-3">{title}</h4>
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm">Naive:</span>
          <span className="font-mono">{naiveValue.toFixed(1)}{unit}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm">Advanced:</span>
          <span className="font-mono font-medium">{advancedValue.toFixed(1)}{unit}</span>
        </div>
        {improvementPercent > 0 && (
          <div className={`flex items-center gap-1 text-xs ${
            isBetter ? 'text-green-600' : 'text-red-600'
          }`}>
            <TrendingUp className={`h-3 w-3 ${!isBetter ? 'rotate-180' : ''}`} />
            {improvementPercent.toFixed(1)}% {isBetter ? 'better' : 'worse'}
          </div>
        )}
      </div>
    </div>
  )
}

export const ComparisonView: React.FC = () => {
  const [currentRequest, setCurrentRequest] = useState<NegotiationRequest | null>(null)
  const { 
    compareRAG, 
    comparisonData, 
    loading, 
    error,
    reset,
    naiveResult,
    advancedResult,
    metrics,
    recommendation
  } = useRAGComparison()
  
  const { apiKeys } = useApiKeyStore()
  const hasRequiredKeys = Boolean(apiKeys.openai)
  const hasEnhancementKeys = Boolean(apiKeys.cohere || apiKeys.exa)

  const handleCompare = async (request: NegotiationRequest) => {
    setCurrentRequest(request)
    await compareRAG(request)
  }

  const handleReset = () => {
    reset()
    setCurrentRequest(null)
  }

  return (
    <div className="w-full max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold">RAG Approach Comparison</h1>
        <p className="text-muted-foreground">
          Compare naive vs advanced RAG approaches for negotiation analysis
        </p>
      </div>

      {/* API Key Warnings */}
      {!hasRequiredKeys && (
        <Alert className="border-red-200 bg-red-50">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Setup Required:</strong> You need to configure your OpenAI API key to use the comparison feature.
          </AlertDescription>
        </Alert>
      )}

      {!hasEnhancementKeys && hasRequiredKeys && (
        <Alert className="border-amber-200 bg-amber-50">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Limited Comparison:</strong> Add Cohere or Exa API keys for the full advanced RAG experience.
          </AlertDescription>
        </Alert>
      )}

      {/* Input Section */}
      {!comparisonData && (
        <MessageInput
          onSubmit={handleCompare}
          loading={loading}
          disabled={!hasRequiredKeys}
        />
      )}

      {/* Loading State */}
      {loading && (
        <div className="text-center py-12">
          <div className="inline-flex items-center gap-3 p-6 bg-card rounded-lg border">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            <div className="text-left">
              <h3 className="font-semibold">Running Comparison...</h3>
              <p className="text-sm text-muted-foreground">
                This may take up to 60 seconds as we analyze with both approaches
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <Alert className="border-red-200 bg-red-50">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Comparison Failed:</strong> {error.message}
          </AlertDescription>
        </Alert>
      )}

      {/* Comparison Results */}
      {comparisonData && naiveResult && advancedResult && metrics && (
        <div className="space-y-8">
          {/* Results Header */}
          <div className="flex items-center justify-between p-4 bg-card rounded-lg border">
            <div className="flex items-center gap-3">
              <Award className="h-5 w-5 text-primary" />
              <h2 className="text-xl font-semibold">Comparison Results</h2>
            </div>
            <Button
              variant="outline"
              onClick={handleReset}
              className="flex items-center gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              New Comparison
            </Button>
          </div>

          {/* Performance Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Response Time"
              naiveValue={metrics.response_time_naive}
              advancedValue={metrics.response_time_advanced}
              unit="s"
              higherIsBetter={false}
            />
            <MetricCard
              title="Accuracy Score"
              naiveValue={metrics.accuracy_score_naive}
              advancedValue={metrics.accuracy_score_advanced}
              unit="%"
            />
            <MetricCard
              title="Relevance Score"
              naiveValue={metrics.relevance_score_naive}
              advancedValue={metrics.relevance_score_advanced}
              unit="%"
            />
            <MetricCard
              title="Confidence Difference"
              naiveValue={naiveResult.confidence}
              advancedValue={advancedResult.confidence}
              unit="%"
            />
          </div>

          {/* Recommendation */}
          {recommendation && (
            <div className="p-6 bg-primary/5 rounded-lg border border-primary/20">
              <div className="flex items-center gap-3 mb-4">
                <Award className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-semibold">Recommendation</h3>
                <Badge className={
                  recommendation.preferred_approach === 'advanced' 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-blue-100 text-blue-800'
                }>
                  {recommendation.preferred_approach} approach preferred
                </Badge>
              </div>
              
              <p className="text-sm mb-4">{recommendation.reasoning}</p>
              
              {recommendation.improvements.length > 0 && (
                <div>
                  <h4 className="font-medium text-sm mb-2">Suggested Improvements:</h4>
                  <ul className="text-sm space-y-1">
                    {recommendation.improvements.map((improvement, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <ArrowRight className="h-3 w-3 mt-1 flex-shrink-0 text-primary" />
                        {improvement}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Side-by-Side Comparison */}
          <div className="grid-cols-1 lg:grid-cols-2 grid gap-8">
            {/* Naive Results */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 p-3 bg-blue-50 rounded-lg border border-blue-200">
                <Zap className="h-5 w-5 text-blue-600" />
                <h3 className="font-semibold text-blue-800">Naive RAG Approach</h3>
                <Badge variant="outline" className="text-blue-700 border-blue-300">
                  Basic
                </Badge>
              </div>
              <ResponseDisplay
                response={naiveResult}
                title="Basic Analysis Results"
              />
            </div>

            {/* Advanced Results */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 p-3 bg-green-50 rounded-lg border border-green-200">
                <Users className="h-5 w-5 text-green-600" />
                <h3 className="font-semibold text-green-800">Advanced RAG Approach</h3>
                <Badge variant="outline" className="text-green-700 border-green-300">
                  Enhanced
                </Badge>
              </div>
              <ResponseDisplay
                response={advancedResult}
                title="Advanced Analysis Results"
              />
            </div>
          </div>

          {/* Original Request Reference */}
          {currentRequest && (
            <div className="p-4 bg-muted/30 rounded-lg border-l-4 border-primary">
              <h4 className="font-medium mb-2">Original Scenario:</h4>
              <p className="text-sm text-muted-foreground mb-3">
                {currentRequest.scenario}
              </p>
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <span>Budget: ${currentRequest.budget?.toLocaleString() || 'N/A'}</span>
                <span>Timeline: {currentRequest.timeline || 'N/A'}</span>
                <span>Client: {currentRequest.client_type || 'N/A'}</span>
                <span>Project: {currentRequest.project_type || 'N/A'}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Welcome State */}
      {!comparisonData && !loading && (
        <div className="text-center py-12 space-y-6">
          <div className="text-6xl">⚖️</div>
          <h3 className="text-xl font-semibold">Compare RAG Approaches</h3>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            See the difference between basic prompting and advanced RAG with reranking, 
            real-time data, and enhanced retrieval techniques.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-2xl mx-auto mt-8">
            <div className="p-6 bg-blue-50 rounded-lg border border-blue-200">
              <Zap className="h-8 w-8 mx-auto mb-3 text-blue-600" />
              <h4 className="font-semibold mb-2 text-blue-800">Naive RAG</h4>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• Basic vector search</li>
                <li>• Standard prompting</li>
                <li>• Faster response time</li>
                <li>• Lower computational cost</li>
              </ul>
            </div>
            
            <div className="p-6 bg-green-50 rounded-lg border border-green-200">
              <Users className="h-8 w-8 mx-auto mb-3 text-green-600" />
              <h4 className="font-semibold mb-2 text-green-800">Advanced RAG</h4>
              <ul className="text-sm text-green-700 space-y-1">
                <li>• Reranked results</li>
                <li>• Real-time market data</li>
                <li>• Enhanced context</li>
                <li>• Higher accuracy</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}