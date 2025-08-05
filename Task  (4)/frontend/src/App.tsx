import { useState } from 'react'
import { Settings, MessageSquare, BarChart3 } from 'lucide-react'
import { Button } from './components/ui/button'
import { ApiKeyManager } from './components/settings'
import { ChatInterface, ComparisonView } from './components/negotiation'

type TabType = 'chat' | 'comparison' | 'settings'

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('chat')

  const tabs = [
    { id: 'chat' as TabType, label: 'Negotiation Chat', icon: MessageSquare },
    { id: 'comparison' as TabType, label: 'RAG Comparison', icon: BarChart3 },
    { id: 'settings' as TabType, label: 'Settings', icon: Settings },
  ]

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="border-b bg-card/50 backdrop-blur supports-[backdrop-filter]:bg-card/60">
        <div className="container mx-auto px-4">
          <div className="flex h-16 items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="text-2xl">🤝</div>
              <h1 className="text-xl font-bold">Freelancer Negotiation Helper</h1>
            </div>
            
            <div className="flex items-center gap-2">
              {tabs.map((tab) => {
                const Icon = tab.icon
                return (
                  <Button
                    key={tab.id}
                    variant={activeTab === tab.id ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setActiveTab(tab.id)}
                    className="flex items-center gap-2"
                  >
                    <Icon className="h-4 w-4" />
                    <span className="hidden sm:inline">{tab.label}</span>
                  </Button>
                )
              })}
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {activeTab === 'chat' && <ChatInterface />}
        {activeTab === 'comparison' && <ComparisonView />}
        {activeTab === 'settings' && (
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold mb-2">API Configuration</h2>
              <p className="text-muted-foreground">
                Configure your API keys to enable negotiation analysis features
              </p>
            </div>
            <ApiKeyManager />
          </div>
        )}
      </main>
    </div>
  )
}

export default App