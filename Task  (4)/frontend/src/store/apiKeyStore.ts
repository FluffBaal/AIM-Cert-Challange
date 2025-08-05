import { create } from 'zustand'

interface ApiKeyStore {
  apiKeys: Record<string, string>
  isInitialized: boolean
  updateApiKey: (keyType: string, value: string) => void
  clearApiKey: (keyType: string) => void
  testApiKey: (keyType: string) => Promise<boolean>
  loadApiKeys: () => void
  clearAllApiKeys: () => void
  getDecryptedKey: (keyType: string) => string | null
}

// Simple encryption for local storage (not for production use)
const encrypt = (text: string): string => {
  return btoa(text)
}

const decrypt = (text: string): string => {
  try {
    return atob(text)
  } catch {
    return ''
  }
}

export const useApiKeyStore = create<ApiKeyStore>((set, get) => ({
  apiKeys: {},
  isInitialized: false,
  
  updateApiKey: (keyType, value) => {
    const encrypted = encrypt(value)
    localStorage.setItem(`api_key_${keyType}`, encrypted)
    set(state => ({
      apiKeys: { ...state.apiKeys, [keyType]: encrypted }
    }))
  },
  
  clearApiKey: (keyType) => {
    localStorage.removeItem(`api_key_${keyType}`)
    set(state => {
      const newKeys = { ...state.apiKeys }
      delete newKeys[keyType]
      return { apiKeys: newKeys }
    })
  },
  
  testApiKey: async (keyType) => {
    const key = get().getDecryptedKey(keyType)
    if (!key) return false
    
    try {
      // Call backend endpoint to test API key
      const response = await fetch('/api/negotiate/test-api-key', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-OpenAI-API-Key': key
        }
      })
      
      return response.ok
    } catch (error) {
      return false
    }
  },
  
  loadApiKeys: () => {
    const keys: Record<string, string> = {}
    const keyTypes = ['openai', 'cohere', 'exa', 'langsmith']
    
    keyTypes.forEach(keyType => {
      const stored = localStorage.getItem(`api_key_${keyType}`)
      if (stored) {
        keys[keyType] = stored
      }
    })
    
    set({ apiKeys: keys, isInitialized: true })
  },
  
  clearAllApiKeys: () => {
    const keyTypes = ['openai', 'cohere', 'exa', 'langsmith']
    keyTypes.forEach(keyType => {
      localStorage.removeItem(`api_key_${keyType}`)
    })
    set({ apiKeys: {} })
  },
  
  getDecryptedKey: (keyType) => {
    const encrypted = get().apiKeys[keyType]
    return encrypted ? decrypt(encrypted) : null
  }
}))

// Initialize on app load
if (typeof window !== 'undefined') {
  useApiKeyStore.getState().loadApiKeys()
}