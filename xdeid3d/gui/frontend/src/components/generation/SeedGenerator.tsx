import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { AdvancedSettings } from './AdvancedSettings'
import { QualityPresetSelector } from './QualityPresetSelector'
import { useGeneration } from '@/hooks/useGeneration'
import { QUALITY_PRESETS, QualityPreset } from '@/types/generation'
import { Dices, Sparkles } from 'lucide-react'

interface SeedGeneratorProps {
  sessionId: string
  isConnected: boolean
  sendMessage: (message: any) => void
}

export function SeedGenerator({ sessionId, isConnected, sendMessage }: SeedGeneratorProps) {
  const { status } = useGeneration()
  const [seed, setSeed] = useState<number>(46)
  const [selectedPreset, setSelectedPreset] = useState<QualityPreset>(QUALITY_PRESETS[1]) // Balanced
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [advancedParams, setAdvancedParams] = useState(QUALITY_PRESETS[1].params)

  const isGenerating = status === 'generating'
  const canGenerate = isConnected && !isGenerating

  const handleRandomSeed = () => {
    setSeed(Math.floor(Math.random() * 1000000))
  }

  const handlePresetChange = (preset: QualityPreset) => {
    setSelectedPreset(preset)
    // Always update advanced params when preset changes
    setAdvancedParams(preset.params)
  }

  const handleGenerate = () => {
    console.log('Generating from seed:', {
      sessionId,
      seed,
      ...advancedParams,
    })

    // Send WebSocket message to backend
    sendMessage({
      type: 'generate_seed',
      seed,
      params: advancedParams,
    })
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Seed Configuration</CardTitle>
          <CardDescription>
            Generate a volumetric identity from a random seed number
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="seed">Seed Number</Label>
            <div className="flex gap-2">
              <Input
                id="seed"
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
                min={0}
                max={999999}
                className="flex-1"
              />
              <Button
                variant="outline"
                size="icon"
                onClick={handleRandomSeed}
                disabled={isGenerating}
              >
                <Dices className="h-4 w-4" />
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Range: 0 - 999,999. Each seed produces a unique identity.
            </p>
          </div>

          <QualityPresetSelector
            selected={selectedPreset}
            onSelect={handlePresetChange}
            disabled={isGenerating}
          />

          <AdvancedSettings
            open={showAdvanced}
            onOpenChange={setShowAdvanced}
            params={advancedParams}
            onParamsChange={setAdvancedParams}
            disabled={isGenerating}
          />
        </CardContent>
      </Card>

      <Alert>
        <Sparkles className="h-4 w-4" />
        <AlertDescription className="text-xs">
          Tip: Lower truncation (0.5-0.6) = higher quality faces. Higher truncation (0.7-0.8) = more diversity. Generates a single frontal identity image (~5-10 seconds).
        </AlertDescription>
      </Alert>

      <Button
        className="w-full"
        size="lg"
        onClick={handleGenerate}
        disabled={!canGenerate}
      >
        {isGenerating ? 'Generating...' : 'Generate Identity Image'}
      </Button>
    </div>
  )
}
