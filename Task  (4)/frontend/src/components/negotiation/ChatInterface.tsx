import React, { useState } from 'react'
import { RefreshCw, Zap, Users, AlertCircle, Target, Brain, DollarSign } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { MessageInput } from './MessageInput'
import { ResponseDisplay } from './ResponseDisplay'
import { useApiClient } from '@/services/api'
import { useApiKeyStore } from '@/store/apiKeyStore'
import { NegotiationRequest, NegotiationResponse } from '@/types/api'

interface ChatSession {
  id: string
  request: NegotiationRequest
  response: NegotiationResponse
  timestamp: Date
  mode: 'naive' | 'advanced'
}

export const ChatInterface: React.FC = () => {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentMode, setCurrentMode] = useState<'naive' | 'advanced'>('advanced')
  const [currentResponse, setCurrentResponse] = useState<NegotiationResponse | null>(null)
  const { negotiate, loading, error } = useApiClient()
  const { apiKeys } = useApiKeyStore()

  // Check if required API keys are present
  const hasRequiredKeys = Boolean(apiKeys.openai)
  const hasEnhancementKeys = Boolean(apiKeys.cohere || apiKeys.exa)

  const handleNegotiate = async (request: NegotiationRequest) => {
    try {
      setCurrentResponse(null)  // Clear previous response
      const response = await negotiate(request, currentMode)
      
      
      
      // Set current response for immediate display
      setCurrentResponse(response)
      
      const newSession: ChatSession = {
        id: crypto.randomUUID(),
        request,
        response,
        timestamp: new Date(),
        mode: currentMode
      }
      
      setSessions(prev => [newSession, ...prev])
    } catch (err) {
      console.error('Negotiation failed:', err)
      setCurrentResponse(null)
      // Error is handled by the useApiClient hook
    }
  }

  const clearHistory = () => {
    setSessions([])
    setCurrentResponse(null)
  }

  const getModeDescription = (mode: 'naive' | 'advanced') => {
    switch (mode) {
      case 'naive':
        return 'Basic analysis using standard prompting'
      case 'advanced':
        return 'Enhanced analysis with RAG and real-time data'
    }
  }

  const getModeColor = (mode: 'naive' | 'advanced') => {
    switch (mode) {
      case 'naive':
        return 'bg-blue-100 text-blue-800'
      case 'advanced':
        return 'bg-green-100 text-green-800'
    }
  }

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold">Freelancer Negotiation Assistant</h1>
        <p className="text-muted-foreground">
          Get expert negotiation advice using Chris Voss techniques and market insights
        </p>
      </div>

      {/* API Key Warning */}
      {!hasRequiredKeys && (
        <Alert className="border-amber-200 bg-amber-50">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Setup Required:</strong> You need to configure your OpenAI API key in the settings below to use the negotiation assistant.
          </AlertDescription>
        </Alert>
      )}

      {/* Mode Selection */}
      <div className="flex items-center justify-between p-4 bg-card rounded-lg border">
        <div className="flex items-center gap-4">
          <span className="font-medium">Analysis Mode:</span>
          <div className="flex gap-2">
            <Button
              variant={currentMode === 'naive' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setCurrentMode('naive')}
              className="flex items-center gap-2"
            >
              <Zap className="h-4 w-4" />
              Naive
            </Button>
            <Button
              variant={currentMode === 'advanced' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setCurrentMode('advanced')}
              disabled={!hasEnhancementKeys}
              className="flex items-center gap-2"
            >
              <Users className="h-4 w-4" />
              Advanced
            </Button>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <Badge className={getModeColor(currentMode)}>
            {getModeDescription(currentMode)}
          </Badge>
          
          {sessions.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={clearHistory}
              className="flex items-center gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              Clear History
            </Button>
          )}
        </div>
      </div>

      {/* Enhancement Keys Warning */}
      {!hasEnhancementKeys && currentMode === 'advanced' && (
        <Alert className="border-blue-200 bg-blue-50">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Enhanced Features:</strong> Add Cohere or Exa API keys in settings to enable advanced RAG capabilities and real-time market data.
          </AlertDescription>
        </Alert>
      )}

      {/* Input Section */}
      <MessageInput
        onSubmit={handleNegotiate}
        loading={loading}
        disabled={!hasRequiredKeys}
      />

      {/* Error Display */}
      {error && (
        <Alert className="border-red-200 bg-red-50">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Error:</strong> {error.message}
          </AlertDescription>
        </Alert>
      )}

      {/* Current Response */}
      {loading && (
        <ResponseDisplay
          response={{} as NegotiationResponse}
          loading={true}
          title="Generating Strategy..."
        />
      )}
      
      {/* Display current response after loading completes */}
      {!loading && currentResponse && (
        <div className="mb-8">
          <ResponseDisplay
            response={currentResponse}
            title={`${currentMode === 'advanced' ? 'Advanced' : 'Basic'} Analysis Results`}
            retrievedContexts={currentResponse.detailed_contexts}
          />
        </div>
      )}

      {/* Session History */}
      {sessions.length > 0 && (
        <div className="space-y-6">
          <h2 className="text-2xl font-semibold border-b pb-2">
            Recent Negotiations ({sessions.length})
          </h2>
          
          {sessions.map((session, index) => (
            <div key={session.id} className="space-y-4">
              {/* Session Header */}
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <Badge variant="outline">#{sessions.length - index}</Badge>
                  <Badge className={getModeColor(session.mode)}>
                    {session.mode}
                  </Badge>
                  <span className="text-sm text-muted-foreground">
                    {session.timestamp.toLocaleString()}
                  </span>
                </div>
                
                <div className="text-sm text-muted-foreground">
                  Budget: ${session.request.budget?.toLocaleString() || 'N/A'} • 
                  Timeline: {session.request.timeline || 'N/A'}
                </div>
              </div>

              {/* Original Request Summary */}
              <div className="p-4 bg-muted/30 rounded-lg border-l-4 border-primary">
                <h4 className="font-medium mb-2">Scenario:</h4>
                <p className="text-sm text-muted-foreground mb-3">
                  {session.request.scenario}
                </p>
                
                {session.request.requirements.length > 0 && (
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-xs font-medium">Requirements:</span>
                    {session.request.requirements.map((req, idx) => (
                      <Badge key={idx} variant="secondary" className="text-xs">
                        {req}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>

              {/* Response */}
              <ResponseDisplay
                response={session.response}
                title={`${session.mode === 'advanced' ? 'Advanced' : 'Basic'} Analysis Results`}
                retrievedContexts={session.response.detailed_contexts}
              />
              
              {index < sessions.length - 1 && (
                <hr className="my-8 border-dashed" />
              )}
            </div>
          ))}
        </div>
      )}

      {/* Welcome Message */}
      {sessions.length === 0 && !loading && (
        <div className="text-center py-12 space-y-4">
          <div className="text-6xl">🤝</div>
          <h3 className="text-xl font-semibold">Ready to Negotiate?</h3>
          <p className="text-muted-foreground max-w-md mx-auto">
            Describe your negotiation scenario above and get expert advice based on proven techniques from Chris Voss and market data.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-2xl mx-auto mt-8">
            <div className="p-4 bg-card rounded-lg border">
              <Target className="h-8 w-8 mx-auto mb-2 text-primary" />
              <h4 className="font-medium mb-1">Strategic Analysis</h4>
              <p className="text-xs text-muted-foreground">
                Get tailored negotiation strategies
              </p>
            </div>
            
            <div className="p-4 bg-card rounded-lg border">
              <Brain className="h-8 w-8 mx-auto mb-2 text-primary" />
              <h4 className="font-medium mb-1">Chris Voss Techniques</h4>
              <p className="text-xs text-muted-foreground">
                Leverage proven FBI negotiation methods
              </p>
            </div>
            
            <div className="p-4 bg-card rounded-lg border">
              <DollarSign className="h-8 w-8 mx-auto mb-2 text-primary" />
              <h4 className="font-medium mb-1">Rate Recommendations</h4>
              <p className="text-xs text-muted-foreground">
                Get market-based pricing advice
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}