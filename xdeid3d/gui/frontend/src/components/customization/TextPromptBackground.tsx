import { useCustomization } from '@/hooks/useCustomization'
import { Card } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Sparkles, Lightbulb } from 'lucide-react'

const PROMPT_EXAMPLES = [
  'Professional photography studio with soft lighting',
  'Modern office environment with glass windows',
  'Outdoor nature scene with bokeh effect',
  'Abstract gradient background with warm colors',
  'Minimalist white studio backdrop',
  'Urban cityscape at golden hour',
]

export function TextPromptBackground() {
  const { backgroundSettings, updateBackgroundSettings } = useCustomization()

  const textPrompt = backgroundSettings.textPrompt || ''
  const negativePrompt = backgroundSettings.negativePrompt || ''

  return (
    <Card className="p-4 space-y-4">
      <div className="space-y-2">
        <Label htmlFor="bg-prompt">Background Description</Label>
        <Textarea
          id="bg-prompt"
          placeholder="Describe the background you want to generate..."
          value={textPrompt}
          onChange={(e) => updateBackgroundSettings({ textPrompt: e.target.value })}
          rows={4}
          className="resize-none"
        />
        <p className="text-xs text-muted-foreground">
          Describe the background environment, lighting, and atmosphere
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="bg-negative">Negative Prompt (Optional)</Label>
        <Textarea
          id="bg-negative"
          placeholder="What to avoid in the background..."
          value={negativePrompt}
          onChange={(e) => updateBackgroundSettings({ negativePrompt: e.target.value })}
          rows={2}
          className="resize-none"
        />
        <p className="text-xs text-muted-foreground">
          Specify elements to exclude from the background
        </p>
      </div>

      <div className="space-y-2">
        <div className="flex items-center gap-2 mb-2">
          <Lightbulb className="h-4 w-4 text-muted-foreground" />
          <Label className="text-xs text-muted-foreground">Example Prompts</Label>
        </div>
        <div className="flex flex-wrap gap-2">
          {PROMPT_EXAMPLES.map((example, index) => (
            <Badge
              key={index}
              variant="outline"
              className="cursor-pointer hover:bg-accent transition-colors text-xs py-1.5 px-2"
              onClick={() => updateBackgroundSettings({ textPrompt: example })}
            >
              {example}
            </Badge>
          ))}
        </div>
      </div>

      {textPrompt && (
        <div className="rounded-lg border p-4 bg-muted/50">
          <div className="flex items-start gap-3">
            <Sparkles className="h-5 w-5 text-primary mt-0.5" />
            <div className="flex-1 space-y-1">
              <p className="text-sm font-medium">AI-Generated Background</p>
              <p className="text-xs text-muted-foreground">
                The background will be generated using Stable Diffusion based on your description.
                This may take a few moments.
              </p>
            </div>
          </div>
        </div>
      )}
    </Card>
  )
}
