import { useCustomization, BackgroundMethod } from '@/hooks/useCustomization'
import { api } from '@/lib/api'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Palette, ImageIcon, Sparkles, Wand2, Loader2 } from 'lucide-react'
import { SolidColorBackground } from './SolidColorBackground'
import { ImageBackground } from './ImageBackground'
import { TextPromptBackground } from './TextPromptBackground'

export function BackgroundSelector() {
  const {
    baseIdentity,
    sessionId,
    backgroundSettings,
    status,
    customizedResult,
    setBackgroundMethod,
    setStatus,
    setProgress,
    setError,
    setCustomizedResult,
  } = useCustomization()

  const handleApplyBackground = async () => {
    if (!baseIdentity || !sessionId) {
      setError('No base identity found. Please generate an identity first.')
      return
    }

    // Validate based on method
    if (backgroundSettings.method === 'color' && !backgroundSettings.solidColor) {
      setError('Please select a color')
      return
    }
    if (backgroundSettings.method === 'image' && !backgroundSettings.imageFile) {
      setError('Please upload an image')
      return
    }
    if (backgroundSettings.method === 'text' && !backgroundSettings.textPrompt?.trim()) {
      setError('Please enter a prompt')
      return
    }

    try {
      setStatus('applying')
      setProgress(0, 'Starting background replacement...')

      let response

      // Extract seed and truncation from metadata (for seed-based generation)
      const seed = baseIdentity.metadata?.seed
      const truncation = baseIdentity.metadata?.params?.truncation || baseIdentity.metadata?.truncation

      console.log('[BackgroundSelector] Applying background with:', {
        sessionId,
        latent_code_path: baseIdentity.latentCodePath,
        seed,
        truncation,
        method: backgroundSettings.method,
      })

      // Call appropriate API based on method
      if (backgroundSettings.method === 'color') {
        setProgress(0.3, 'Applying color background...')
        response = await api.applyColorBackground({
          session_id: sessionId,
          latent_code_path: baseIdentity.latentCodePath,
          seed: seed,
          truncation: truncation,
          color: backgroundSettings.solidColor!,
          generator_path: baseIdentity.generatorPath,
        })
      } else if (backgroundSettings.method === 'image') {
        setProgress(0.3, 'Applying image background...')
        response = await api.applyImageBackground({
          session_id: sessionId,
          latent_code_path: baseIdentity.latentCodePath,
          seed: seed,
          truncation: truncation,
          image_fit: backgroundSettings.imageFit || 'cover',
          background_image: backgroundSettings.imageFile!,
          generator_path: baseIdentity.generatorPath,
        })
      } else if (backgroundSettings.method === 'text') {
        setProgress(0.3, 'Generating background from prompt...')
        response = await api.applyTextBackground({
          session_id: sessionId,
          latent_code_path: baseIdentity.latentCodePath,
          seed: seed,
          truncation: truncation,
          prompt: backgroundSettings.textPrompt!,
          negative_prompt: backgroundSettings.negativePrompt,
          generator_path: baseIdentity.generatorPath,
        })
      }

      if (response && response.success) {
        console.log('[BackgroundSelector] Response:', response)
        console.log('[BackgroundSelector] Image URL:', response.image_url)

        // Update result (image_url is at top level in CustomizationResponse)
        // This automatically sets status to 'complete' and progress to 1
        setCustomizedResult({
          imageUrl: response.image_url || '',
          metadata: {
            method: backgroundSettings.method,
            settings: backgroundSettings,
          },
        })

        console.log('[BackgroundSelector] Set customized result with imageUrl:', response.image_url)
      } else {
        const errorMsg = response?.error || 'Failed to apply background'
        console.error('[BackgroundSelector] Error response:', errorMsg)
        setError(errorMsg)
      }
    } catch (err) {
      console.error('[BackgroundSelector] Exception:', err)
      const errorMsg = err instanceof Error ? err.message : 'An error occurred while applying background'
      setError(errorMsg)
    }
  }

  const canApply = () => {
    switch (backgroundSettings.method) {
      case 'color':
        return !!backgroundSettings.solidColor
      case 'image':
        return !!backgroundSettings.imageFile || !!backgroundSettings.imageUrl
      case 'text':
        return !!backgroundSettings.textPrompt && backgroundSettings.textPrompt.trim().length > 0
      default:
        return false
    }
  }

  const getButtonText = () => {
    if (status === 'applying') return 'Applying...'
    if (customizedResult) return 'Re-apply Background'
    return 'Apply Background'
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-medium mb-2">Background Type</h3>
        <p className="text-xs text-muted-foreground mb-4">
          Choose how to customize the background
        </p>
      </div>

      <Tabs
        value={backgroundSettings.method}
        onValueChange={(v) => setBackgroundMethod(v as BackgroundMethod)}
        className="w-full"
      >
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="color" className="gap-1.5">
            <Palette className="h-3.5 w-3.5" />
            Color
          </TabsTrigger>
          <TabsTrigger value="image" className="gap-1.5">
            <ImageIcon className="h-3.5 w-3.5" />
            Image
          </TabsTrigger>
          <TabsTrigger value="text" className="gap-1.5">
            <Sparkles className="h-3.5 w-3.5" />
            Text
          </TabsTrigger>
        </TabsList>

        <TabsContent value="color" className="mt-4">
          <SolidColorBackground />
        </TabsContent>

        <TabsContent value="image" className="mt-4">
          <ImageBackground />
        </TabsContent>

        <TabsContent value="text" className="mt-4">
          <TextPromptBackground />
        </TabsContent>
      </Tabs>

      <Separator />

      {/* Apply Button */}
      <div className="space-y-3">
        <Button
          className="w-full"
          size="lg"
          onClick={handleApplyBackground}
          disabled={!canApply() || status === 'applying'}
        >
          {status === 'applying' ? (
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <Wand2 className="h-4 w-4 mr-2" />
          )}
          {getButtonText()}
        </Button>

        {!canApply() && status !== 'applying' && (
          <p className="text-xs text-center text-muted-foreground">
            {backgroundSettings.method === 'color' && 'Select a color to continue'}
            {backgroundSettings.method === 'image' && 'Upload an image to continue'}
            {backgroundSettings.method === 'text' && 'Enter a description to continue'}
          </p>
        )}
      </div>
    </div>
  )
}
