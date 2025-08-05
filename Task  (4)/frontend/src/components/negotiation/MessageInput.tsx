import React, { useState } from 'react'
import { Send, DollarSign, Clock, FileText, Building, FolderOpen } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { NegotiationRequest } from '@/types/api'

interface MessageInputProps {
  onSubmit: (request: NegotiationRequest) => void
  loading?: boolean
  disabled?: boolean
}

const CLIENT_TYPES = [
  'startup', 'enterprise', 'agency', 'non-profit', 'government', 'individual'
]

const PROJECT_TYPES = [
  'web-development', 'mobile-app', 'data-analysis', 'consulting', 
  'design', 'marketing', 'research', 'other'
]

export const MessageInput: React.FC<MessageInputProps> = ({
  onSubmit,
  loading = false,
  disabled = false
}) => {
  const [formData, setFormData] = useState<NegotiationRequest>({
    scenario: '',
    budget: 0,
    timeline: '',
    requirements: [],
    client_type: '',
    project_type: ''
  })
  
  const [newRequirement, setNewRequirement] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!formData.scenario.trim()) return
    onSubmit(formData)
  }

  const addRequirement = () => {
    if (newRequirement.trim() && !formData.requirements.includes(newRequirement.trim())) {
      setFormData(prev => ({
        ...prev,
        requirements: [...prev.requirements, newRequirement.trim()]
      }))
      setNewRequirement('')
    }
  }

  const removeRequirement = (requirement: string) => {
    setFormData(prev => ({
      ...prev,
      requirements: prev.requirements.filter(req => req !== requirement)
    }))
  }

  const handleKeyPress = (e: React.KeyboardEvent, action: () => void) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      action()
    }
  }

  const isFormValid = formData.scenario.trim().length > 0

  return (
    <div className="w-full max-w-4xl mx-auto">
      <form onSubmit={handleSubmit} className="space-y-6 p-6 bg-card rounded-lg border">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Scenario Input */}
          <div className="md:col-span-2">
            <Label htmlFor="scenario" className="flex items-center gap-2 text-base font-medium">
              <FileText className="h-4 w-4" />
              Negotiation Scenario
            </Label>
            <textarea
              id="scenario"
              value={formData.scenario}
              onChange={(e) => setFormData(prev => ({ ...prev, scenario: e.target.value }))}
              placeholder="Describe your negotiation situation... (e.g., 'Client wants to reduce my proposed rate for a 3-month web development project...')"
              className="mt-2 w-full min-h-[120px] p-3 border rounded-md resize-none focus:ring-2 focus:ring-primary focus:border-primary"
              disabled={disabled || loading}
            />
          </div>

          {/* Budget */}
          <div>
            <Label htmlFor="budget" className="flex items-center gap-2 text-base font-medium">
              <DollarSign className="h-4 w-4" />
              Your Proposed Budget ($)
            </Label>
            <Input
              id="budget"
              type="number"
              value={formData.budget || ''}
              onChange={(e) => setFormData(prev => ({ ...prev, budget: Number(e.target.value) }))}
              placeholder="5000"
              min="0"
              step="100"
              className="mt-2"
              disabled={disabled || loading}
            />
          </div>

          {/* Timeline */}
          <div>
            <Label htmlFor="timeline" className="flex items-center gap-2 text-base font-medium">
              <Clock className="h-4 w-4" />
              Project Timeline
            </Label>
            <Input
              id="timeline"
              value={formData.timeline}
              onChange={(e) => setFormData(prev => ({ ...prev, timeline: e.target.value }))}
              placeholder="3 months"
              className="mt-2"
              disabled={disabled || loading}
            />
          </div>

          {/* Client Type */}
          <div>
            <Label className="flex items-center gap-2 text-base font-medium">
              <Building className="h-4 w-4" />
              Client Type
            </Label>
            <div className="mt-2 flex flex-wrap gap-2">
              {CLIENT_TYPES.map((type) => (
                <Button
                  key={type}
                  type="button"
                  variant={formData.client_type === type ? "default" : "outline"}
                  size="sm"
                  onClick={() => setFormData(prev => ({ ...prev, client_type: type }))}
                  disabled={disabled || loading}
                  className="capitalize"
                >
                  {type}
                </Button>
              ))}
            </div>
          </div>

          {/* Project Type */}
          <div>
            <Label className="flex items-center gap-2 text-base font-medium">
              <FolderOpen className="h-4 w-4" />
              Project Type
            </Label>
            <div className="mt-2 flex flex-wrap gap-2">
              {PROJECT_TYPES.map((type) => (
                <Button
                  key={type}
                  type="button"
                  variant={formData.project_type === type ? "default" : "outline"}
                  size="sm"
                  onClick={() => setFormData(prev => ({ ...prev, project_type: type }))}
                  disabled={disabled || loading}
                  className="capitalize"
                >
                  {type.replace('-', ' ')}
                </Button>
              ))}
            </div>
          </div>

          {/* Requirements */}
          <div className="md:col-span-2">
            <Label htmlFor="requirements" className="text-base font-medium">
              Key Requirements/Deliverables
            </Label>
            <div className="mt-2 flex gap-2">
              <Input
                id="requirements"
                value={newRequirement}
                onChange={(e) => setNewRequirement(e.target.value)}
                placeholder="Add a requirement..."
                onKeyPress={(e) => handleKeyPress(e, addRequirement)}
                disabled={disabled || loading}
              />
              <Button
                type="button"
                variant="outline"
                onClick={addRequirement}
                disabled={!newRequirement.trim() || disabled || loading}
              >
                Add
              </Button>
            </div>
            
            {formData.requirements.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {formData.requirements.map((req, index) => (
                  <Badge
                    key={index}
                    variant="secondary"
                    className="cursor-pointer hover:bg-destructive hover:text-destructive-foreground"
                    onClick={() => removeRequirement(req)}
                  >
                    {req} ×
                  </Badge>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex justify-end pt-4 border-t">
          <Button
            type="submit"
            disabled={!isFormValid || disabled || loading}
            className="flex items-center gap-2 px-6"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current"></div>
                Analyzing...
              </>
            ) : (
              <>
                <Send className="h-4 w-4" />
                Get Negotiation Advice
              </>
            )}
          </Button>
        </div>
      </form>
    </div>
  )
}