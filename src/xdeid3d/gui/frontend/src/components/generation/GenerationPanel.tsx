import { useState, useCallback } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import { SeedGenerator } from './SeedGenerator'
import { ImageUploader } from './ImageUploader'
import { TextPromptGenerator } from './TextPromptGenerator'
import { PreviewPanel } from './PreviewPanel'
import { useGeneration } from '@/hooks/useGeneration'
import { useWebSocket } from '@/hooks/useWebSocket'
import { api } from '@/lib/api'
import { generateSessionId } from '@/lib/utils'
import { Dices, Upload, Pencil } from 'lucide-react'

interface GenerationPanelProps {
  onProceedToCustomization?: () => void
}

export function GenerationPanel({ onProceedToCustomization }: GenerationPanelProps) {
  const [sessionId] = useState(() => generateSessionId())
  const { method, setMethod, handleWebSocketMessage } = useGeneration()

  // Stable callback functions to prevent WebSocket reconnection
  const onConnect = useCallback(() => {
    console.log('WebSocket connected')
  }, [])

  const onDisconnect = useCallback(() => {
    console.log('WebSocket disconnected')
  }, [])

  const onError = useCallback((error: Event) => {
    console.error('WebSocket error:', error)
  }, [])

  const wsUrl = api.getWebSocketUrl(sessionId)
  const { isConnected, sendMessage } = useWebSocket(wsUrl, {
    onMessage: handleWebSocketMessage,
    onConnect,
    onDisconnect,
    onError,
  })

  return (
    <div className="h-full flex">
      {/* Left Panel - Generation Input */}
      <div className="w-[420px] border-r flex flex-col">
        <div className="p-6 flex-1 overflow-y-auto">
          <div className="mb-4">
            <h2 className="text-lg font-semibold mb-1">Generate Identity</h2>
            <p className="text-sm text-muted-foreground">
              Create a 3D volumetric identity using one of three methods
            </p>
          </div>

          <Separator className="my-4" />

          <Tabs value={method} onValueChange={(v) => setMethod(v as any)} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="seed" className="gap-1.5">
                <Dices className="h-4 w-4" />
                Seed
              </TabsTrigger>
              <TabsTrigger value="upload" className="gap-1.5">
                <Upload className="h-4 w-4" />
                Upload
              </TabsTrigger>
              <TabsTrigger value="text" className="gap-1.5">
                <Pencil className="h-4 w-4" />
                Text
              </TabsTrigger>
            </TabsList>

            <TabsContent value="seed" className="mt-4">
              <SeedGenerator sessionId={sessionId} isConnected={isConnected} sendMessage={sendMessage} />
            </TabsContent>

            <TabsContent value="upload" className="mt-4">
              <ImageUploader sessionId={sessionId} isConnected={isConnected} sendMessage={sendMessage} />
            </TabsContent>

            <TabsContent value="text" className="mt-4">
              <TextPromptGenerator sessionId={sessionId} isConnected={isConnected} sendMessage={sendMessage} />
            </TabsContent>
          </Tabs>
        </div>

        {/* Connection Status */}
        <div className="px-6 py-3 border-t">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            {isConnected ? 'Connected to server' : 'Connecting...'}
          </div>
        </div>
      </div>

      {/* Right Panel - Preview */}
      <div className="flex-1 overflow-hidden">
        <PreviewPanel onProceedToCustomization={onProceedToCustomization} />
      </div>
    </div>
  )
}
