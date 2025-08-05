import React, { useState } from 'react'
import {
  ChevronDown,
  ChevronRight,
  Database,
  Brain,
  FileText,
  CheckCircle,
  Clock,
  Loader2,
  Info,
  Zap,
  Globe,
  BookOpen,
  Target,
  Sparkles,
  Trophy,
  Rocket,
  Star,
  PartyPopper
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { NegotiationResponse } from '@/types/api'

interface AnalysisTransparencyProps {
  response: NegotiationResponse
  loading?: boolean
  processingSteps?: ProcessingStep[]
}

export interface ProcessingStep {
  id: string
  name: string
  description: string
  status: 'pending' | 'active' | 'completed' | 'skipped'
  icon: React.ElementType
  startTime?: number
  endTime?: number
  details?: any
}

const getIconForStep = (stepId: string): React.ElementType => {
  const iconMap: Record<string, React.ElementType> = {
    'retrieval': Database,
    'context_analysis': Brain,
    'market_search': Globe,
    'strategy': Target,
    'response': FileText
  }
  return iconMap[stepId] || Brain
}

const getFunCompletionIcon = (stepId: string): React.ElementType => {
  // Fun icons for completed steps
  const funIconMap: Record<string, React.ElementType> = {
    'retrieval': Sparkles,
    'context_analysis': Star,
    'market_search': Rocket,
    'strategy': Trophy,
    'response': PartyPopper
  }
  return funIconMap[stepId] || CheckCircle
}

const defaultSteps: ProcessingStep[] = [
  {
    id: 'retrieval',
    name: 'Knowledge Retrieval',
    description: 'Searching negotiation tactics database',
    status: 'pending',
    icon: Database
  },
  {
    id: 'context_analysis',
    name: 'Context Analysis',
    description: 'Understanding your negotiation scenario',
    status: 'pending',
    icon: Brain
  },
  {
    id: 'market_search',
    name: 'Market Research',
    description: 'Fetching real-time rate data',
    status: 'pending',
    icon: Globe
  },
  {
    id: 'strategy',
    name: 'Strategy Formation',
    description: 'Applying Chris Voss techniques',
    status: 'pending',
    icon: Target
  },
  {
    id: 'response',
    name: 'Response Generation',
    description: 'Crafting your negotiation response',
    status: 'pending',
    icon: FileText
  }
]

export const AnalysisTransparency: React.FC<AnalysisTransparencyProps> = ({
  response,
  loading = false,
  processingSteps
}) => {
  const [isOpen, setIsOpen] = useState(true)
  const [selectedStep, setSelectedStep] = useState<string | null>(null)
  const [simulatedProgress, setSimulatedProgress] = useState(0)
  
  // Simulate progress during loading
  React.useEffect(() => {
    if (loading) {
      setSimulatedProgress(0)
      const interval = setInterval(() => {
        setSimulatedProgress(prev => {
          if (prev >= 90) return prev // Stop at 90% until real completion
          return prev + Math.random() * 10
        })
      }, 1000)
      return () => clearInterval(interval)
    } else {
      setSimulatedProgress(100)
    }
  }, [loading])
  
  // Use backend steps if available, otherwise use defaults with simulated progress
  const steps = React.useMemo(() => {
    if (processingSteps && processingSteps.length > 0 && !loading) {
      return processingSteps.map(step => ({
        ...step,
        icon: getIconForStep(step.id)
      }))
    }
    
    // During loading, simulate progress
    if (loading) {
      const progressPerStep = simulatedProgress / defaultSteps.length
      return defaultSteps.map((step, index) => {
        const stepProgress = progressPerStep * (index + 1)
        let status: ProcessingStep['status'] = 'pending'
        
        if (stepProgress > 90) {
          status = 'active'
        } else if (stepProgress > 50) {
          status = index < 2 ? 'completed' : 'active'
        } else if (stepProgress > 20) {
          status = index === 0 ? 'completed' : index === 1 ? 'active' : 'pending'
        } else if (stepProgress > 0) {
          status = index === 0 ? 'active' : 'pending'
        }
        
        return {
          ...step,
          status
        }
      })
    }
    
    return defaultSteps
  }, [processingSteps, loading, simulatedProgress])

  const getStepIcon = (step: ProcessingStep) => {
    const Icon = step.icon
    
    // Check if all steps are completed for fun icons
    const allCompleted = !loading && steps.every(s => s.status === 'completed' || s.status === 'skipped')
    
    if (step.status === 'completed') {
      if (allCompleted) {
        // Use fun completion icons when everything is done
        const FunIcon = getFunCompletionIcon(step.id)
        return <FunIcon className="h-5 w-5 text-green-600 animate-bounce" />
      }
      return <CheckCircle className="h-5 w-5 text-green-600" />
    }
    if (step.status === 'active') return <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />
    if (step.status === 'skipped') return <Icon className="h-5 w-5 text-gray-400" />
    return <Clock className="h-5 w-5 text-gray-400" />
  }

  const getStepTime = (step: ProcessingStep) => {
    if (step.startTime && step.endTime) {
      return `${((step.endTime - step.startTime) / 1000).toFixed(1)}s`
    }
    return null
  }

  const getProgressPercentage = () => {
    if (loading) {
      return simulatedProgress
    }
    
    const completed = steps.filter(s => s.status === 'completed').length
    const active = steps.filter(s => s.status === 'active').length
    const skipped = steps.filter(s => s.status === 'skipped').length
    const total = steps.length
    
    if (total === 0) return 0
    
    return ((completed + skipped + (active * 0.5)) / total) * 100
  }

  return (
    <div className="w-full max-w-4xl mx-auto mb-6">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <div className="border rounded-lg bg-card">
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center justify-between p-4 hover:bg-muted/50 transition-colors">
              <div className="flex items-center gap-3">
                <Zap className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-semibold">Analysis Process</h3>
                {loading && (
                  <Badge variant="outline" className="animate-pulse">
                    <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                    Processing
                  </Badge>
                )}
                {!loading && steps.every(s => s.status === 'completed' || s.status === 'skipped') && (
                  <Badge className="bg-gradient-to-r from-green-500 to-emerald-500 text-white border-0">
                    <Sparkles className="h-3 w-3 mr-1" />
                    Complete!
                  </Badge>
                )}
              </div>
              {isOpen ? <ChevronDown className="h-5 w-5" /> : <ChevronRight className="h-5 w-5" />}
            </div>
          </CollapsibleTrigger>

          <CollapsibleContent>
            <div className="px-4 pb-4">
              {/* Progress Bar */}
              {(loading || (processingSteps && processingSteps.length > 0)) && (
                <div className="mb-4">
                  <Progress value={getProgressPercentage()} className="h-2" />
                  {loading && (
                    <p className="text-xs text-muted-foreground mt-1">
                      Processing your negotiation analysis...
                    </p>
                  )}
                </div>
              )}

              {/* Processing Steps */}
              <div className="space-y-3">
                {steps.map((step) => (
                  <div
                    key={step.id}
                    className={`p-3 rounded-lg border transition-all cursor-pointer ${
                      selectedStep === step.id ? 'border-primary bg-primary/5' : 
                      step.status === 'completed' && !loading && steps.every(s => s.status === 'completed' || s.status === 'skipped') ? 
                      'border-green-300 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/10 dark:to-emerald-950/10' :
                      'hover:bg-muted/50'
                    }`}
                    onClick={() => setSelectedStep(selectedStep === step.id ? null : step.id)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        {getStepIcon(step)}
                        <div>
                          <p className="font-medium">{step.name}</p>
                          <p className="text-sm text-muted-foreground">{step.description}</p>
                        </div>
                      </div>
                      {getStepTime(step) && (
                        <Badge variant="secondary" className="text-xs">
                          {getStepTime(step)}
                        </Badge>
                      )}
                    </div>

                    {/* Step Details */}
                    {selectedStep === step.id && step.details && (
                      <div className="mt-3 pt-3 border-t">
                        {step.id === 'retrieval' && response.retrieval_details && (
                          <div className="space-y-2 text-sm">
                            <p className="flex items-center gap-2">
                              <Database className="h-4 w-4" />
                              Mode: <Badge variant="outline">{response.mode}</Badge>
                            </p>
                            <p>Documents searched: {response.retrieval_details.documents_searched || 'N/A'}</p>
                            <p>Chunks retrieved: {response.retrieval_details.chunks_retrieved || 'N/A'}</p>
                            <p>Relevance threshold: {response.retrieval_details.relevance_threshold || '0.7'}</p>
                          </div>
                        )}

                        {step.id === 'market_search' && response.market_data && (
                          <div className="space-y-2 text-sm">
                            <p className="flex items-center gap-2">
                              <Globe className="h-4 w-4" />
                              Data points found: {response.market_data.data_points}
                            </p>
                            <p>Sources: {response.market_data.sources?.length || 0} websites</p>
                            {response.market_data.sources && response.market_data.sources[0] !== 'fallback_estimates' && (
                              <div className="flex flex-wrap gap-1 mt-2">
                                {response.market_data.sources.slice(0, 3).map((source, idx) => (
                                  <Badge key={idx} variant="secondary" className="text-xs">
                                    {source.includes('upwork') ? 'Upwork' : 
                                     source.includes('glassdoor') ? 'Glassdoor' :
                                     source.includes('indeed') ? 'Indeed' : 'Web'}
                                  </Badge>
                                ))}
                              </div>
                            )}
                          </div>
                        )}

                        {step.id === 'strategy' && response.strategy && (
                          <div className="space-y-2 text-sm">
                            <p className="flex items-center gap-2">
                              <BookOpen className="h-4 w-4" />
                              Techniques applied: {response.strategy.techniques.length}
                            </p>
                            <div className="flex flex-wrap gap-1 mt-2">
                              {response.strategy.techniques.map((technique, idx) => (
                                <Badge key={idx} variant="secondary" className="text-xs">
                                  {technique.replace(/_/g, ' ')}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Agents Used */}
              {response.agents_used && response.agents_used.length > 0 && (
                <div className="mt-4 p-3 bg-muted/50 rounded-lg">
                  <p className="text-sm font-medium mb-2 flex items-center gap-2">
                    <Brain className="h-4 w-4" />
                    AI Agents Involved
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {response.agents_used.map((agent, idx) => (
                      <Badge key={idx} variant="outline">
                        {agent}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Success Celebration Message */}
              {!loading && steps.every(s => s.status === 'completed' || s.status === 'skipped') && (
                <div className="mt-4 p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20 rounded-lg border border-green-200 dark:border-green-800">
                  <p className="text-sm font-semibold mb-2 flex items-center gap-2 text-green-800 dark:text-green-200">
                    <Trophy className="h-5 w-5 text-yellow-500 animate-pulse" />
                    Analysis Complete! 🎉
                  </p>
                  <p className="text-sm text-green-700 dark:text-green-300">
                    Your negotiation strategy has been crafted using advanced AI analysis, Chris Voss techniques, 
                    and real-time market data. You're ready to negotiate like a pro!
                  </p>
                </div>
              )}

              {/* Confidence Explanation */}
              {!loading && (
                <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800">
                  <p className="text-sm font-medium mb-1 flex items-center gap-2">
                    <Info className="h-4 w-4 text-blue-600" />
                    Confidence Score: {Math.round(response.confidence * 100)}%
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {response.confidence >= 0.8 
                      ? "High confidence - Strong pattern match with successful negotiation tactics"
                      : response.confidence >= 0.6
                      ? "Good confidence - Solid application of negotiation principles"
                      : "Moderate confidence - General negotiation strategies applied"}
                  </p>
                </div>
              )}
            </div>
          </CollapsibleContent>
        </div>
      </Collapsible>
    </div>
  )
}