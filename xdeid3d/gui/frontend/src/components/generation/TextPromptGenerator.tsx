import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Separator } from '@/components/ui/separator'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { QualityPresetSelector } from './QualityPresetSelector'
import { useGeneration } from '@/hooks/useGeneration'
import { QUALITY_PRESETS, QualityPreset } from '@/types/generation'
import { Sparkles, Lightbulb, Settings2, ChevronDown, ChevronUp, Info } from 'lucide-react'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

interface TextPromptGeneratorProps {
  sessionId: string
  isConnected: boolean
  sendMessage: (message: any) => void
}

const PROMPT_EXAMPLES = [
  'A professional portrait of a person with short hair',
  'A person with long brown hair and blue eyes',
  'A smiling person with glasses',
  'A person with facial hair and friendly expression',
]

export function TextPromptGenerator({ sessionId, isConnected, sendMessage }: TextPromptGeneratorProps) {
  const { status } = useGeneration()
  const [prompt, setPrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('blurry, low quality, distorted, multiple faces')
  const [steps, setSteps] = useState([50])
  const [guidanceScale, setGuidanceScale] = useState([7.5])

  // PTI settings
  const [selectedPreset, setSelectedPreset] = useState<QualityPreset>(QUALITY_PRESETS[1]) // Balanced
  const [showAdvancedPTI, setShowAdvancedPTI] = useState(false)
  const [wSteps, setWSteps] = useState(500)
  const [ptiSteps, setPtiSteps] = useState(350)

  // Noise optimization settings
  const [optimizeNoise, setOptimizeNoise] = useState(false)
  const [initialNoiseFactor, setInitialNoiseFactor] = useState(0.05)
  const [noiseRampLength, setNoiseRampLength] = useState(0.75)
  const [regularizeNoiseWeight, setRegularizeNoiseWeight] = useState(100000) // 1e5

  const isGenerating = status === 'generating'
  const canGenerate = isConnected && !isGenerating && prompt.trim().length > 0

  const handleExampleClick = (example: string) => {
    setPrompt(example)
  }

  const handlePresetChange = (preset: QualityPreset) => {
    setSelectedPreset(preset)
    // Update PTI steps when preset changes
    if (preset.ptiSteps) {
      setWSteps(preset.ptiSteps.wSteps)
      setPtiSteps(preset.ptiSteps.ptiSteps)
    }
  }

  const handleGenerate = () => {
    console.log('Generating from text:', {
      sessionId,
      prompt,
      negativePrompt,
      sdParams: {
        steps: steps[0],
        guidanceScale: guidanceScale[0],
      },
      ptiParams: {
        wSteps,
        ptiSteps,
        truncation: selectedPreset.params.truncation,
        nrr: selectedPreset.params.nrr,
        sampleMult: selectedPreset.params.sampleMult,
        optimizeNoise,
        initialNoiseFactor,
        noiseRampLength,
        regularizeNoiseWeight,
      },
    })

    // Send WebSocket message to backend with both SD and PTI parameters
    sendMessage({
      type: 'generate_text',
      prompt,
      negative_prompt: negativePrompt,
      steps: steps[0],
      guidance_scale: guidanceScale[0],
      // PTI parameters for Stage 2
      pti_params: {
        w_steps: wSteps,
        pti_steps: ptiSteps,
        truncation: selectedPreset.params.truncation,
        nrr: selectedPreset.params.nrr,
        sampleMult: selectedPreset.params.sampleMult,
        optimize_noise: optimizeNoise,
        initial_noise_factor: initialNoiseFactor,
        noise_ramp_length: noiseRampLength,
        regularize_noise_weight: regularizeNoiseWeight,
      },
    })
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Text-to-Image Generation</CardTitle>
          <CardDescription>
            Describe the face you want to generate
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="prompt">Prompt</Label>
            <Textarea
              id="prompt"
              placeholder="Describe the face you want to generate..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              disabled={isGenerating}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="negative-prompt">Negative Prompt</Label>
            <Textarea
              id="negative-prompt"
              placeholder="What to avoid in generation..."
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              rows={2}
              disabled={isGenerating}
            />
          </div>

          <div className="space-y-3">
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Steps</Label>
                <span className="text-sm text-muted-foreground">{steps[0]}</span>
              </div>
              <Slider
                value={steps}
                onValueChange={setSteps}
                min={20}
                max={100}
                step={10}
                disabled={isGenerating}
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Guidance Scale</Label>
                <span className="text-sm text-muted-foreground">{guidanceScale[0]}</span>
              </div>
              <Slider
                value={guidanceScale}
                onValueChange={setGuidanceScale}
                min={5}
                max={15}
                step={0.5}
                disabled={isGenerating}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Lightbulb className="h-4 w-4" />
            <CardTitle className="text-sm">Example Prompts</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-2">
          {PROMPT_EXAMPLES.map((example, i) => (
            <button
              key={i}
              onClick={() => handleExampleClick(example)}
              className="w-full text-left px-3 py-2 text-sm rounded-md border hover:bg-accent transition-colors"
              disabled={isGenerating}
            >
              {example}
            </button>
          ))}
        </CardContent>
      </Card>

      {/* PTI Settings Card */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">3D Conversion Settings (PTI)</CardTitle>
          <CardDescription>
            Configure quality for Stage 2 (image to 3D)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <QualityPresetSelector
            selected={selectedPreset}
            onSelect={handlePresetChange}
            disabled={isGenerating}
          />

          <Separator />

          {/* Advanced PTI Settings */}
          <div className="space-y-3">
            <Button
              variant="ghost"
              className="w-full justify-between h-auto py-2"
              onClick={() => setShowAdvancedPTI(!showAdvancedPTI)}
              disabled={isGenerating}
            >
              <div className="flex items-center gap-2">
                <Settings2 className="h-4 w-4" />
                <span className="font-medium">Advanced PTI Settings</span>
              </div>
              {showAdvancedPTI ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>

            {showAdvancedPTI && (
              <div className="space-y-4 pt-2">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label className="text-sm">W-Projection Steps</Label>
                    <span className="text-sm text-muted-foreground">{wSteps}</span>
                  </div>
                  <Slider
                    value={[wSteps]}
                    onValueChange={(v) => setWSteps(v[0])}
                    min={100}
                    max={3000}
                    step={100}
                    disabled={isGenerating}
                  />
                  <p className="text-xs text-muted-foreground">
                    More steps = better quality, longer time (~{Math.round(wSteps / 100)} min)
                  </p>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label className="text-sm">PTI Fine-tuning Steps</Label>
                    <span className="text-sm text-muted-foreground">{ptiSteps}</span>
                  </div>
                  <Slider
                    value={[ptiSteps]}
                    onValueChange={(v) => setPtiSteps(v[0])}
                    min={100}
                    max={2000}
                    step={100}
                    disabled={isGenerating}
                  />
                  <p className="text-xs text-muted-foreground">
                    More steps = better reconstruction (~{Math.round(ptiSteps / 100)} min)
                  </p>
                </div>
              </div>
            )}
          </div>

          <Separator />

          {/* Noise Optimization Section */}
          <TooltipProvider>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Label htmlFor="optimize-noise-text" className="text-sm font-medium">
                    Noise Optimization
                  </Label>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      <p className="text-xs">
                        Enables optimization of noise inputs for finer details and textures.
                        This can improve quality but increases projection time by ~20-30%.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </div>
                <Switch
                  id="optimize-noise-text"
                  checked={optimizeNoise}
                  onCheckedChange={setOptimizeNoise}
                  disabled={isGenerating}
                />
              </div>

              {optimizeNoise && (
                <div className="space-y-4 pl-4 border-l-2 border-primary/20">
                  {/* Initial Noise Factor */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Label className="text-sm">Initial Noise Factor</Label>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Info className="h-3 w-3 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent className="max-w-xs">
                            <p className="text-xs">
                              Starting strength of noise injection. Higher values add more variation
                              but may reduce stability. Default: 0.05
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                      <span className="text-sm text-muted-foreground">{initialNoiseFactor.toFixed(3)}</span>
                    </div>
                    <Slider
                      value={[initialNoiseFactor * 1000]}
                      onValueChange={(v) => setInitialNoiseFactor(v[0] / 1000)}
                      min={10}
                      max={200}
                      step={5}
                      disabled={isGenerating}
                    />
                    <p className="text-xs text-muted-foreground">
                      Range: 0.01 - 0.20 (lower = more stable, higher = more detail variation)
                    </p>
                  </div>

                  {/* Noise Ramp Length */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Label className="text-sm">Noise Ramp Length</Label>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Info className="h-3 w-3 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent className="max-w-xs">
                            <p className="text-xs">
                              Controls how long noise persists during optimization (as fraction of total steps).
                              Longer = more detail exploration. Default: 0.75 (75% of steps)
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                      <span className="text-sm text-muted-foreground">{(noiseRampLength * 100).toFixed(0)}%</span>
                    </div>
                    <Slider
                      value={[noiseRampLength * 100]}
                      onValueChange={(v) => setNoiseRampLength(v[0] / 100)}
                      min={25}
                      max={100}
                      step={5}
                      disabled={isGenerating}
                    />
                    <p className="text-xs text-muted-foreground">
                      Percentage of optimization steps with noise active
                    </p>
                  </div>

                  {/* Noise Regularization Weight */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Label className="text-sm">Noise Regularization</Label>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Info className="h-3 w-3 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent className="max-w-xs">
                            <p className="text-xs">
                              Strength of noise smoothness constraint. Higher values = smoother noise patterns,
                              lower values = more detailed but potentially noisy. Default: 100k (1e5)
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                      <span className="text-sm text-muted-foreground">{(regularizeNoiseWeight / 1000).toFixed(0)}k</span>
                    </div>
                    <Slider
                      value={[regularizeNoiseWeight / 1000]}
                      onValueChange={(v) => setRegularizeNoiseWeight(v[0] * 1000)}
                      min={10}
                      max={500}
                      step={10}
                      disabled={isGenerating}
                    />
                    <p className="text-xs text-muted-foreground">
                      Range: 10k - 500k (lower = more detail, higher = smoother)
                    </p>
                  </div>
                </div>
              )}
            </div>
          </TooltipProvider>
        </CardContent>
      </Card>

      <Alert>
        <Sparkles className="h-4 w-4" />
        <AlertDescription className="text-xs">
          <strong>Two-stage process:</strong> Stage 1: Generate face image with Stable Diffusion (~1-2 min). Stage 2: Convert to 3D using PTI (~{Math.round((wSteps + ptiSteps) / 100)} min). Total: ~{Math.round((wSteps + ptiSteps) / 100) + 2} min.
        </AlertDescription>
      </Alert>

      <Button
        className="w-full"
        size="lg"
        onClick={handleGenerate}
        disabled={!canGenerate}
      >
        {isGenerating ? 'Generating...' : 'Generate from Text'}
      </Button>
    </div>
  )
}
