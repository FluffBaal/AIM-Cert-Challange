import * as React from "react"
import { useState, useEffect } from "react"
import { Eye, EyeOff, Loader2, Check, X } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "./button"
import { Input } from "./input"
import { Label } from "./label"
import { Badge } from "./badge"

interface ApiKeyInputProps {
  keyType: string
  label: string
  placeholder: string
  value: string
  onChange: (value: string) => void
  onTest?: () => void
  onClear?: () => void
  required?: boolean
  helpText?: string
  validateFormat?: (key: string) => boolean
  isTestingKey?: boolean
  testResult?: 'success' | 'error' | null
}

export const ApiKeyInput: React.FC<ApiKeyInputProps> = ({
  keyType,
  label,
  placeholder,
  value,
  onChange,
  onTest,
  onClear,
  required = false,
  helpText,
  validateFormat,
  isTestingKey = false,
  testResult
}) => {
  const [showKey, setShowKey] = useState(false)
  const [localValue, setLocalValue] = useState(value)
  const [isValid, setIsValid] = useState(true)
  
  useEffect(() => {
    setLocalValue(value)
  }, [value])
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value
    setLocalValue(newValue)
    
    // Validate format if validator provided
    if (validateFormat && newValue) {
      setIsValid(validateFormat(newValue))
    }
  }
  
  const handleBlur = () => {
    if (localValue !== value) {
      onChange(localValue)
    }
  }
  
  
  return (
    <div className="api-key-input-container">
      <div className="flex items-center justify-between mb-2">
        <Label htmlFor={`api-key-${keyType}`} className="flex items-center gap-2">
          {label}
          {required && <span className="text-destructive">*</span>}
        </Label>
        {value && (
          <div className="flex items-center gap-2">
            {testResult === 'success' && (
              <Badge variant="success" className="text-xs">
                <Check className="w-3 h-3 mr-1" />
                Verified
              </Badge>
            )}
            {testResult === 'error' && (
              <Badge variant="destructive" className="text-xs">
                <X className="w-3 h-3 mr-1" />
                Invalid
              </Badge>
            )}
          </div>
        )}
      </div>
      
      <div className="relative">
        <Input
          id={`api-key-${keyType}`}
          type={showKey ? 'text' : 'password'}
          placeholder={placeholder}
          value={localValue}
          onChange={handleChange}
          onBlur={handleBlur}
          className={cn(
            'pr-24',
            !isValid && localValue && 'border-destructive focus:ring-destructive'
          )}
        />
        
        <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setShowKey(!showKey)}
            className="h-7 w-7 p-0"
          >
            {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          </Button>
          
          {onTest && value && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={onTest}
              disabled={isTestingKey || !isValid}
              className="h-7 px-2"
            >
              {isTestingKey ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                'Test'
              )}
            </Button>
          )}
          
          {onClear && value && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={onClear}
              className="h-7 w-7 p-0"
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
      
      {helpText && (
        <p className="text-xs text-muted-foreground mt-1">{helpText}</p>
      )}
      
      {!isValid && localValue && (
        <p className="text-xs text-destructive mt-1">
          Invalid format. Expected format: {placeholder}
        </p>
      )}
    </div>
  )
}