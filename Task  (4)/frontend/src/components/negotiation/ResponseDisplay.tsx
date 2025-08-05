import React from 'react'
import { 
  Target, 
  Shield, 
  CheckCircle, 
  AlertTriangle,
  DollarSign,
  MessageSquare,
  Brain
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { NegotiationResponse } from '@/types/api'
import { AnalysisTransparency, ProcessingStep } from './AnalysisTransparency'
import { RetrievedContext, RetrievedChunk } from './RetrievedContext'

interface ResponseDisplayProps {
  response: NegotiationResponse
  loading?: boolean
  title?: string
  processingSteps?: ProcessingStep[]
  retrievedContexts?: RetrievedChunk[]
}

// Chris Voss negotiation technique categories with colors
const CHRIS_VOSS_TECHNIQUES = {
  mirroring: { 
    color: 'bg-blue-100 text-blue-800 border-blue-200', 
    icon: MessageSquare,
    description: 'Repeat key words to build rapport'
  },
  labeling: { 
    color: 'bg-green-100 text-green-800 border-green-200', 
    icon: Target,
    description: 'Identify and acknowledge emotions'
  },
  tactical_empathy: { 
    color: 'bg-purple-100 text-purple-800 border-purple-200', 
    icon: Brain,
    description: 'Understand their perspective'
  },
  calibrated_questions: { 
    color: 'bg-orange-100 text-orange-800 border-orange-200', 
    icon: CheckCircle,
    description: 'How/What questions for control'
  },
  anchoring: { 
    color: 'bg-red-100 text-red-800 border-red-200', 
    icon: DollarSign,
    description: 'Set initial reference point'
  },
  loss_aversion: { 
    color: 'bg-yellow-100 text-yellow-800 border-yellow-200', 
    icon: Shield,
    description: 'Highlight what they might lose'
  }
}

const getConfidenceColor = (score: number) => {
  if (score >= 80) return 'text-green-600 bg-green-50'
  if (score >= 60) return 'text-yellow-600 bg-yellow-50'
  return 'text-red-600 bg-red-50'
}

const getConfidenceIcon = (score: number) => {
  if (score >= 80) return CheckCircle
  if (score >= 60) return AlertTriangle
  return AlertTriangle
}


export const ResponseDisplay: React.FC<ResponseDisplayProps> = ({
  response,
  loading = false,
  title = "Negotiation Strategy",
  processingSteps,
  retrievedContexts
}) => {
  
  const ConfidenceIcon = getConfidenceIcon(response?.confidence || 0)

  if (loading) {
    return (
      <div className="w-full max-w-4xl mx-auto space-y-6">
        {/* Show Analysis Transparency during loading */}
        <AnalysisTransparency 
          response={{} as NegotiationResponse}
          loading={true}
          processingSteps={processingSteps}
        />
        
        <div className="p-6 bg-card rounded-lg border">
          <div className="flex items-center gap-2 mb-6">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary"></div>
            <h2 className="text-xl font-semibold">Analyzing your negotiation...</h2>
          </div>
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Our AI agents are working on your negotiation strategy. This typically takes 30-60 seconds.
            </p>
            <div className="bg-muted/50 p-4 rounded-lg">
              <p className="text-sm font-medium mb-2">What's happening:</p>
              <ul className="text-sm text-muted-foreground space-y-1 list-disc list-inside">
                <li>Searching negotiation tactics from "Never Split the Difference"</li>
                <li>Analyzing your specific context and requirements</li>
                <li>Researching current market rates for your skill</li>
                <li>Crafting a personalized negotiation strategy</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6">
      {/* Analysis Transparency Component */}
      <AnalysisTransparency 
        response={response}
        loading={loading}
        processingSteps={processingSteps}
      />

      {/* Retrieved Context Component */}
      {retrievedContexts && retrievedContexts.length > 0 && (
        <RetrievedContext 
          contexts={retrievedContexts}
          loading={loading}
        />
      )}
      {/* Header with Confidence Score */}
      <div className="p-6 bg-gradient-to-r from-primary/10 to-primary/5 rounded-lg border">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="h-6 w-6" />
            {title}
          </h2>
          <div className={`flex items-center gap-2 px-3 py-1 rounded-full ${getConfidenceColor(response.confidence)}`}>
            <ConfidenceIcon className="h-4 w-4" />
            <span className="font-medium">{Math.round(response.confidence * 100)}% confidence</span>
          </div>
        </div>
        
        {/* Quick Summary */}
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div className="text-center p-3 bg-background/50 rounded-lg">
            <p className="text-xs text-muted-foreground uppercase tracking-wide">Approach</p>
            <p className="text-sm font-semibold mt-1 capitalize">{response.strategy.approach}</p>
          </div>
          <div className="text-center p-3 bg-background/50 rounded-lg">
            <p className="text-xs text-muted-foreground uppercase tracking-wide">Techniques</p>
            <p className="text-sm font-semibold mt-1">{response.strategy.techniques.length} Applied</p>
          </div>
          <div className="text-center p-3 bg-background/50 rounded-lg">
            <p className="text-xs text-muted-foreground uppercase tracking-wide">Analysis Mode</p>
            <p className="text-sm font-semibold mt-1 capitalize">{response.mode}</p>
          </div>
        </div>
        
        {/* Recommended Response */}
        <div>
          <h3 className="text-sm font-semibold text-muted-foreground mb-2 uppercase tracking-wide">
            Recommended Response
          </h3>
          <div className="p-4 bg-background border-l-4 border-primary rounded-r-lg shadow-sm">
            <p className="text-base leading-relaxed italic">"{response.rewritten_text}"</p>
          </div>
        </div>
      </div>

      {/* Market Data (if available) - Moved up for prominence */}
      {response?.market_data && (
        <div className="p-6 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20 rounded-lg border border-green-200 dark:border-green-800">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <DollarSign className="h-5 w-5 text-green-600" />
            Real-Time Market Rate Analysis
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-900 p-3 rounded-lg">
              <p className="text-sm text-muted-foreground">Min Rate</p>
              <p className="text-xl font-bold text-green-600">${Math.round(response.market_data.min_rate)}/hr</p>
            </div>
            <div className="bg-white dark:bg-gray-900 p-3 rounded-lg">
              <p className="text-sm text-muted-foreground">Max Rate</p>
              <p className="text-xl font-bold text-green-600">${Math.round(response.market_data.max_rate)}/hr</p>
            </div>
            <div className="bg-white dark:bg-gray-900 p-3 rounded-lg border-2 border-green-400">
              <p className="text-sm text-muted-foreground">Median Rate</p>
              <p className="text-xl font-bold text-green-600">${Math.round(response.market_data.median_rate)}/hr</p>
            </div>
            <div className="bg-white dark:bg-gray-900 p-3 rounded-lg">
              <p className="text-sm text-muted-foreground">Data Points</p>
              <p className="text-xl font-bold">{response.market_data.data_points}</p>
            </div>
          </div>
          
          {/* Show sources if real data */}
          {response.market_data.sources && response.market_data.sources.length > 0 && 
           !response.market_data.sources.includes('fallback_estimates') && (
            <div className="mt-4">
              <p className="text-sm font-semibold text-muted-foreground mb-2">Data Sources:</p>
              <div className="flex flex-wrap gap-2">
                {response.market_data.sources.slice(0, 3).map((source, idx) => {
                  const domain = source.includes('upwork') ? 'Upwork' : 
                               source.includes('glassdoor') ? 'Glassdoor' :
                               source.includes('indeed') ? 'Indeed' :
                               source.includes('freelancer') ? 'Freelancer' : 'Web';
                  return (
                    <Badge key={idx} variant="secondary" className="text-xs">
                      {domain}
                    </Badge>
                  );
                })}
                {response.market_data.sources.length > 3 && (
                  <Badge variant="secondary" className="text-xs">
                    +{response.market_data.sources.length - 3} more
                  </Badge>
                )}
              </div>
            </div>
          )}
          
          {response.market_data.note && (
            <p className="text-sm text-muted-foreground mt-4 italic">{response.market_data.note}</p>
          )}
        </div>
      )}

      {/* Chris Voss Techniques Used */}
      <div className="p-6 bg-card rounded-lg border">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Brain className="h-5 w-5" />
          Negotiation Techniques Applied
        </h3>
        <div className="flex flex-wrap gap-2 mb-4">
          {response.strategy.techniques.map((technique, index) => {
            const techniqueConfig = CHRIS_VOSS_TECHNIQUES[technique as keyof typeof CHRIS_VOSS_TECHNIQUES]
            const TechniqueIcon = techniqueConfig?.icon
            
            return (
              <Badge key={index} className={techniqueConfig?.color || 'bg-gray-100 text-gray-800'}>
                {TechniqueIcon && <TechniqueIcon className="h-3 w-3 mr-1" />}
                {technique.replace(/_/g, ' ').toUpperCase()}
              </Badge>
            )
          })}
        </div>
        <p className="text-sm text-muted-foreground">
          Approach: <span className="font-medium">{response.strategy.approach}</span>
        </p>
      </div>

      {/* Strategic Analysis */}
      <div className="p-6 bg-card rounded-lg border">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Target className="h-5 w-5" />
          Strategic Analysis
        </h3>
        
        {/* Key Insights */}
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-3 uppercase tracking-wide">
            Key Insights
          </h4>
          <div className="space-y-2">
            {response.insights.map((insight, index) => (
              <div key={index} className="flex items-start gap-3">
                <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-foreground">{insight}</p>
              </div>
            ))}
            {/* Add market insight if available */}
            {response.market_data && response.market_data.sources && 
             !response.market_data.sources.includes('fallback_estimates') && (
              <div className="flex items-start gap-3">
                <DollarSign className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-foreground">
                  Real-time market analysis from {response.market_data.data_points} data points shows median rates 
                  of ${Math.round(response.market_data.median_rate)}/hr in your market
                </p>
              </div>
            )}
          </div>
        </div>
        
        {/* Situation Assessment */}
        <div className="grid md:grid-cols-2 gap-4">
          <div className="p-4 bg-orange-50 dark:bg-orange-950/20 rounded-lg border border-orange-200 dark:border-orange-800">
            <h4 className="font-medium text-orange-800 dark:text-orange-300 mb-2 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              Current Situation
            </h4>
            <p className="text-sm text-orange-700 dark:text-orange-400">
              {response.mode === 'fallback' 
                ? 'Using general negotiation strategies. Add more context for specific advice.'
                : 'Analysis based on Chris Voss techniques and best practices.'}
            </p>
          </div>
          
          <div className="p-4 bg-green-50 dark:bg-green-950/20 rounded-lg border border-green-200 dark:border-green-800">
            <h4 className="font-medium text-green-800 dark:text-green-300 mb-2 flex items-center gap-2">
              <Shield className="h-4 w-4" />
              Strategy Confidence
            </h4>
            <p className="text-sm text-green-700 dark:text-green-400">
              {response.confidence >= 0.7 
                ? 'High confidence - Strong match with proven techniques'
                : response.confidence >= 0.5
                ? 'Moderate confidence - General principles apply'
                : 'Low confidence - Consider additional context'}
            </p>
          </div>
        </div>
      </div>



      {/* Next Steps */}
      <div className="p-6 bg-card rounded-lg border border-primary/20 bg-primary/5">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <MessageSquare className="h-5 w-5" />
          Recommended Next Steps
        </h3>
        <div className="space-y-3">
          <div className="flex items-start gap-3">
            <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary text-primary-foreground text-xs font-bold flex-shrink-0">1</span>
            <div>
              <p className="font-medium text-sm">Send the recommended response</p>
              <p className="text-sm text-muted-foreground">Use the crafted message above to maintain negotiation momentum</p>
            </div>
          </div>
          
          <div className="flex items-start gap-3">
            <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary text-primary-foreground text-xs font-bold flex-shrink-0">2</span>
            <div>
              <p className="font-medium text-sm">Listen for their counter</p>
              <p className="text-sm text-muted-foreground">Apply active listening and look for underlying concerns</p>
            </div>
          </div>
          
          <div className="flex items-start gap-3">
            <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary text-primary-foreground text-xs font-bold flex-shrink-0">3</span>
            <div>
              <p className="font-medium text-sm">Be prepared to pivot</p>
              <p className="text-sm text-muted-foreground">Have alternative solutions ready based on their feedback</p>
            </div>
          </div>
        </div>
        
        {response.market_data && (
          <div className="mt-4 p-3 bg-background rounded-lg">
            <p className="text-sm font-medium mb-1">💡 Pricing Guidance</p>
            <p className="text-sm text-muted-foreground">
              Based on market data, consider a range between ${response.market_data.min_rate} - ${response.market_data.median_rate} 
              as your negotiation zone.
            </p>
          </div>
        )}
      </div>

      {/* Performance Metrics */}
      <div className="p-6 bg-card rounded-lg border">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Shield className="h-5 w-5" />
          Performance Metrics
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div>
            <p className="text-sm text-muted-foreground">Confidence</p>
            <p className="text-lg font-semibold">{Math.round(response.confidence * 100)}%</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Mode</p>
            <p className="text-lg font-semibold capitalize">{response.mode}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Retrieval Time</p>
            <p className="text-lg font-semibold">{response.retrieval_time.toFixed(2)}s</p>
          </div>
        </div>
        {response.partial_failure && (
          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
            <p className="text-sm text-yellow-800">
              <AlertTriangle className="h-4 w-4 inline mr-1" />
              Some features were unavailable. Using fallback data.
            </p>
          </div>
        )}
      </div>

      {/* Chris Voss Techniques Legend */}
      <div className="p-6 bg-muted/30 rounded-lg border border-dashed">
        <h4 className="font-medium mb-3 flex items-center gap-2">
          <MessageSquare className="h-4 w-4" />
          Chris Voss Negotiation Techniques Reference
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {Object.entries(CHRIS_VOSS_TECHNIQUES).map(([key, config]) => {
            const Icon = config.icon
            return (
              <div key={key} className="flex items-center gap-2 text-xs">
                <Badge className={config.color}>
                  <Icon className="h-3 w-3 mr-1" />
                  {key.replace('_', ' ').toUpperCase()}
                </Badge>
                <span className="text-muted-foreground">{config.description}</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}