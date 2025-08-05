import React, { useState } from 'react'
import {
  BookOpen,
  FileText,
  Star,
  ChevronDown,
  ChevronRight,
  Hash,
  Quote
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Card } from '@/components/ui/card'

interface RetrievedContextProps {
  contexts?: RetrievedChunk[]
  loading?: boolean
}

export interface RetrievedChunk {
  content: string
  source: string
  chapter?: string
  page?: number
  relevance_score: number
  technique?: string
  metadata?: Record<string, any>
}

export const RetrievedContext: React.FC<RetrievedContextProps> = ({
  contexts = [],
  loading = false
}) => {
  const [isOpen, setIsOpen] = useState(false)
  const [expandedChunks, setExpandedChunks] = useState<Set<number>>(new Set())

  const toggleChunk = (index: number) => {
    const newExpanded = new Set(expandedChunks)
    if (newExpanded.has(index)) {
      newExpanded.delete(index)
    } else {
      newExpanded.add(index)
    }
    setExpandedChunks(newExpanded)
  }

  const getRelevanceColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50'
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-50'
    return 'text-orange-600 bg-orange-50'
  }

  const getRelevanceLabel = (score: number) => {
    if (score >= 0.8) return 'High'
    if (score >= 0.6) return 'Good'
    return 'Fair'
  }

  if (loading || contexts.length === 0) {
    return null
  }

  return (
    <div className="w-full max-w-4xl mx-auto mb-6">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <Card>
          <CollapsibleTrigger className="w-full">
            <div className="flex items-center justify-between p-4 hover:bg-muted/50 transition-colors">
              <div className="flex items-center gap-3">
                <BookOpen className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-semibold">Knowledge Sources Used</h3>
                <Badge variant="secondary">{contexts.length} sources</Badge>
              </div>
              {isOpen ? <ChevronDown className="h-5 w-5" /> : <ChevronRight className="h-5 w-5" />}
            </div>
          </CollapsibleTrigger>

          <CollapsibleContent>
            <div className="px-4 pb-4">
              <p className="text-sm text-muted-foreground mb-4">
                These are the most relevant excerpts from "Never Split the Difference" that informed your negotiation strategy:
              </p>

              <div className="space-y-3">
                {contexts.map((chunk, index) => (
                  <div
                    key={index}
                    className="border rounded-lg p-4 hover:border-primary/50 transition-colors"
                  >
                    {/* Header */}
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-2 flex-wrap">
                        <FileText className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm font-medium">{chunk.source}</span>
                        {chunk.chapter && (
                          <Badge variant="outline" className="text-xs">
                            Chapter {chunk.chapter}
                          </Badge>
                        )}
                        {chunk.technique && (
                          <Badge className="text-xs">
                            {chunk.technique.replace(/_/g, ' ')}
                          </Badge>
                        )}
                      </div>
                      <div className={`px-2 py-1 rounded-full text-xs font-medium ${getRelevanceColor(chunk.relevance_score)}`}>
                        <Star className="h-3 w-3 inline mr-1" />
                        {getRelevanceLabel(chunk.relevance_score)} ({Math.round(chunk.relevance_score * 100)}%)
                      </div>
                    </div>

                    {/* Content */}
                    <div className="relative">
                      <Quote className="absolute -top-1 -left-1 h-4 w-4 text-muted-foreground/30" />
                      <div className={`pl-6 ${!expandedChunks.has(index) ? 'line-clamp-3' : ''}`}>
                        <p className="text-sm text-foreground/90 italic">
                          {chunk.content}
                        </p>
                      </div>
                      
                      {chunk.content.length > 200 && (
                        <button
                          onClick={() => toggleChunk(index)}
                          className="mt-2 text-sm text-primary hover:underline flex items-center gap-1"
                        >
                          {expandedChunks.has(index) ? (
                            <>Show less <ChevronDown className="h-3 w-3" /></>
                          ) : (
                            <>Show more <ChevronRight className="h-3 w-3" /></>
                          )}
                        </button>
                      )}
                    </div>

                    {/* Metadata */}
                    {chunk.metadata && Object.keys(chunk.metadata).length > 0 && (
                      <div className="mt-3 pt-3 border-t flex flex-wrap gap-2">
                        {Object.entries(chunk.metadata).map(([key, value]) => (
                          <div key={key} className="flex items-center gap-1 text-xs text-muted-foreground">
                            <Hash className="h-3 w-3" />
                            <span>{key}: {String(value)}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Summary */}
              <div className="mt-4 p-3 bg-muted/50 rounded-lg">
                <p className="text-sm font-medium mb-1">Retrieval Summary</p>
                <p className="text-sm text-muted-foreground">
                  Retrieved {contexts.length} relevant passages with an average relevance score of{' '}
                  {Math.round(contexts.reduce((acc, c) => acc + c.relevance_score, 0) / contexts.length * 100)}%.
                  These sources provided specific negotiation tactics and examples that were adapted to your situation.
                </p>
              </div>
            </div>
          </CollapsibleContent>
        </Card>
      </Collapsible>
    </div>
  )
}