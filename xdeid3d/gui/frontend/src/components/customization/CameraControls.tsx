import { useCustomization } from '@/hooks/useCustomization'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Video, Grid3x3 } from 'lucide-react'
import { KeyframeTimeline } from './KeyframeTimeline'
import { PosePreviewSettings } from './PosePreviewSettings'
import { useEffect, useRef } from 'react'
import { api } from '@/lib/api'

const CAMERA_PRESETS = [
  {
    id: 'rotate360',
    name: '360° Rotation',
    icon: '🔄',
    description: 'Full horizontal rotation around the head',
  },
  {
    id: 'orbit',
    name: 'Orbital',
    icon: '🌍',
    description: 'Circular orbit with pitch variation',
  },
  {
    id: 'sidebyside',
    name: 'Side-to-Side',
    icon: '↔️',
    description: 'Left-right oscillation',
  },
  {
    id: 'front',
    name: 'Front View',
    icon: '👤',
    description: 'Static frontal view',
  },
]

export function CameraControls() {
  const {
    baseIdentity,
    sessionId,
    cameraPose,
    setCameraPoseMode,
    setPreset,
    posePreview,
    setPosePreviewStatus,
    setPosePreviewUrl,
    setPosePreviewError,
  } = useCustomization()

  // Debounce timer ref
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null)

  // Track previous state to detect changes
  const prevStateRef = useRef({
    preset: cameraPose.preset,
    keyframes: JSON.stringify(cameraPose.keyframes),
    frames: posePreview.settings.frames,
    fps: posePreview.settings.fps,
    quality: posePreview.settings.quality,
  })

  const handlePresetSelect = (presetId: string) => {
    setPreset(presetId)
  }

  const triggerPreviewRender = async () => {
    if (!baseIdentity || !sessionId) {
      console.log('[CameraControls] No base identity or session ID')
      return
    }

    // Check if we can generate preview
    const canRender = cameraPose.mode === 'preset'
      ? !!cameraPose.preset
      : cameraPose.keyframes.length >= 2

    if (!canRender) {
      console.log('[CameraControls] Cannot render preview - insufficient data')
      return
    }

    try {
      setPosePreviewStatus('rendering', 0, 'Initializing preview rendering...')

      console.log('[CameraControls] Rendering preview:', {
        mode: cameraPose.mode,
        preset: cameraPose.preset,
        keyframes: cameraPose.keyframes.length,
        settings: posePreview.settings,
      })

      const response = await api.renderPosePreview({
        session_id: sessionId,
        mode: cameraPose.mode,
        preset_id: cameraPose.preset || undefined,
        keyframes: cameraPose.mode === 'keyframes'
          ? cameraPose.keyframes.map(kf => ({
              timestamp: kf.timestamp,
              yaw: kf.yaw,
              pitch: kf.pitch,
              radius: kf.radius,
            }))
          : undefined,
        interpolation: cameraPose.interpolation,
        duration_seconds: cameraPose.duration, // Use timeline duration
        preview_fps: posePreview.settings.fps,
        preview_quality: posePreview.settings.quality,
        latent_code_path: baseIdentity.latentCodePath,
        generator_path: baseIdentity.generatorPath,
        seed: baseIdentity.metadata?.seed,
        truncation: baseIdentity.metadata?.params?.truncation || baseIdentity.metadata?.truncation || 0.65,
      })

      if (response.success && response.preview_url) {
        console.log('[CameraControls] Preview rendered successfully:', response.preview_url)
        setPosePreviewUrl(response.preview_url)
      } else {
        throw new Error(response.error || 'Failed to render preview')
      }
    } catch (error) {
      console.error('[CameraControls] Preview rendering failed:', error)
      setPosePreviewError(error instanceof Error ? error.message : 'Failed to render preview')
    }
  }

  // Auto-trigger preview when settings change (with debounce)
  useEffect(() => {
    const currentState = {
      preset: cameraPose.preset,
      keyframes: JSON.stringify(cameraPose.keyframes),
      frames: posePreview.settings.frames,
      fps: posePreview.settings.fps,
      quality: posePreview.settings.quality,
    }

    // Check if anything changed
    const hasChanged = Object.keys(currentState).some(
      key => currentState[key as keyof typeof currentState] !== prevStateRef.current[key as keyof typeof prevStateRef.current]
    )

    if (hasChanged) {
      console.log('[CameraControls] Settings changed, scheduling preview render')

      // Clear existing timer
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
      }

      // Set new timer (500ms debounce)
      debounceTimerRef.current = setTimeout(() => {
        console.log('[CameraControls] Debounce timeout - triggering preview render')
        triggerPreviewRender()
      }, 500)

      // Update prev state
      prevStateRef.current = currentState
    }

    // Cleanup on unmount
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
      }
    }
  }, [
    cameraPose.preset,
    cameraPose.keyframes,
    cameraPose.mode,
    posePreview.settings.frames,
    posePreview.settings.fps,
    posePreview.settings.quality,
    baseIdentity,
    sessionId,
  ])

  return (
    <div className="space-y-4">
      <Tabs
        value={cameraPose.mode}
        onValueChange={(v) => setCameraPoseMode(v as 'preset' | 'keyframes')}
      >
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="preset" className="gap-2">
            <Grid3x3 className="h-3.5 w-3.5" />
            Presets
          </TabsTrigger>
          <TabsTrigger value="keyframes" className="gap-2">
            <Video className="h-3.5 w-3.5" />
            Keyframes
          </TabsTrigger>
        </TabsList>

        <TabsContent value="preset" className="mt-4 space-y-3">
          <div className="grid grid-cols-2 gap-2">
            {CAMERA_PRESETS.map((preset) => (
              <Card
                key={preset.id}
                className={`p-3 cursor-pointer transition-all hover:border-primary ${
                  cameraPose.preset === preset.id
                    ? 'border-primary bg-primary/10'
                    : 'border-border'
                }`}
                onClick={() => handlePresetSelect(preset.id)}
              >
                <div className="text-2xl mb-2">{preset.icon}</div>
                <h4 className="text-sm font-medium mb-1">{preset.name}</h4>
                <p className="text-xs text-muted-foreground line-clamp-2">
                  {preset.description}
                </p>
              </Card>
            ))}
          </div>

          {cameraPose.preset && (
            <div className="text-xs text-muted-foreground">
              Selected: <span className="font-medium text-foreground">
                {CAMERA_PRESETS.find(p => p.id === cameraPose.preset)?.name}
              </span>
            </div>
          )}
        </TabsContent>

        <TabsContent value="keyframes" className="mt-4">
          <KeyframeTimeline />
        </TabsContent>
      </Tabs>

      <Separator />

      {/* Preview Settings */}
      <PosePreviewSettings />
    </div>
  )
}
