import React, { useState } from "react"
import { AlertCircle } from "lucide-react"

import { useApiKeyStore } from "@/store/apiKeyStore"
import { ApiKeyInput } from "@/components/ui/api-key-input"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

interface ApiKeyConfig {
  key: string
  label: string
  placeholder: string
  required: boolean
  helpText: string
  validateFormat: (key: string) => boolean
}

const API_KEY_CONFIGS: Record<string, ApiKeyConfig> = {
  openai: {
    key: 'openai',
    label: 'OpenAI API Key',
    placeholder: 'sk-...',
    required: true,
    helpText: 'Required for core negotiation analysis',
    validateFormat: (key) => key.startsWith('sk-') && key.length > 20
  },
  cohere: {
    key: 'cohere',
    label: 'Cohere API Key',
    placeholder: 'Enter your Cohere API key',
    required: false,
    helpText: 'Optional: Improves retrieval quality with reranking',
    validateFormat: (key) => key.length > 10
  },
  exa: {
    key: 'exa',
    label: 'Exa AI API Key',
    placeholder: 'Enter your Exa AI API key',
    required: false,
    helpText: 'Optional: Enables real-time market data search',
    validateFormat: (key) => key.length > 10
  },
  langsmith: {
    key: 'langsmith',
    label: 'LangSmith API Key',
    placeholder: 'Enter your LangSmith API key',
    required: false,
    helpText: 'Optional: Enables performance monitoring',
    validateFormat: (key) => key.length > 10
  }
}

export const ApiKeyManager: React.FC = () => {
  const { apiKeys, updateApiKey, testApiKey, clearApiKey, getDecryptedKey } = useApiKeyStore()
  const [testingKey, setTestingKey] = useState<string | null>(null)
  const [testResults, setTestResults] = useState<Record<string, 'success' | 'error' | null>>({})
  
  const handleTestApiKey = async (keyType: string) => {
    setTestingKey(keyType)
    try {
      const result = await testApiKey(keyType)
      setTestResults(prev => ({ ...prev, [keyType]: result ? 'success' : 'error' }))
    } catch (error) {
      setTestResults(prev => ({ ...prev, [keyType]: 'error' }))
    } finally {
      setTestingKey(null)
      // Clear test result after 3 seconds
      setTimeout(() => {
        setTestResults(prev => ({ ...prev, [keyType]: null }))
      }, 3000)
    }
  }
  
  const getMissingRequiredKeys = () => {
    return Object.values(API_KEY_CONFIGS)
      .filter(config => config.required && !apiKeys[config.key])
      .map(config => config.label)
  }
  
  const missingKeys = getMissingRequiredKeys()
  
  return (
    <div className="api-key-manager">
      {missingKeys.length > 0 && (
        <Alert className="mb-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Required API Keys Missing</AlertTitle>
          <AlertDescription>
            Please add the following required API keys: {missingKeys.join(', ')}
          </AlertDescription>
        </Alert>
      )}
      
      <div className="space-y-4">
        {Object.values(API_KEY_CONFIGS).map((config) => (
          <ApiKeyInput
            key={config.key}
            keyType={config.key}
            label={config.label}
            placeholder={config.placeholder}
            value={getDecryptedKey(config.key) || ''}
            onChange={(value) => updateApiKey(config.key, value)}
            onTest={() => handleTestApiKey(config.key)}
            onClear={() => clearApiKey(config.key)}
            required={config.required}
            helpText={config.helpText}
            validateFormat={config.validateFormat}
            isTestingKey={testingKey === config.key}
            testResult={testResults[config.key]}
          />
        ))}
      </div>
      
      <div className="mt-6 p-4 bg-muted rounded-lg">
        <h4 className="font-medium mb-2">Security Note</h4>
        <p className="text-sm text-muted-foreground">
          API keys are stored locally in your browser and are never sent to our servers. 
          They are only transmitted directly to the respective API services.
        </p>
      </div>
    </div>
  )
}