import { useState, useEffect } from 'react'
import { useGeneration } from '@/hooks/useGeneration'
import { useCustomization } from '@/hooks/useCustomization'
import { Card } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { LivePreviewImage } from './LivePreviewImage'
import { formatProgress } from '@/lib/utils'
import {
  Loader2,
  CheckCircle2,
  XCircle,
  Image as ImageIcon,
  Download,
  ArrowRight,
  Film,
  ImageIcon as ImageResultIcon,
} from 'lucide-react'

interface PreviewPanelProps {
  onProceedToCustomization?: () => void
}

export function PreviewPanel({ onProceedToCustomization }: PreviewPanelProps) {
  const { status, progress, stage, currentStep, totalSteps, message, previewImage, result, error, method } =
    useGeneration()
  const { setBaseIdentity } = useCustomization()
  const [viewMode, setViewMode] = useState<'result' | 'progress'>('result')
  const [videoLoaded, setVideoLoaded] = useState(false)

  const handleProceedToCustomization = () => {
    if (result) {
      // Generate a customization session ID
      const customizationSessionId = `custom-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
      setBaseIdentity(result, customizationSessionId)
      onProceedToCustomization?.()
    }
  }

  const getStatusInfo = () => {
    switch (status) {
      case 'idle':
        return {
          title: 'Ready to Generate',
          description: 'Select a generation method and configure parameters to begin',
          icon: <ImageIcon className="h-12 w-12 text-muted-foreground" />,
        }
      case 'generating':
        return {
          title: 'Generating...',
          description: message || 'Processing your request',
          icon: <Loader2 className="h-12 w-12 text-primary animate-spin" />,
        }
      case 'complete':
        return {
          title: 'Generation Complete',
          description: 'Your 3D head model is ready',
          icon: <CheckCircle2 className="h-12 w-12 text-green-500" />,
        }
      case 'error':
        return {
          title: 'Generation Failed',
          description: error || 'An error occurred during generation',
          icon: <XCircle className="h-12 w-12 text-destructive" />,
        }
    }
  }

  const statusInfo = getStatusInfo()

  const getStageLabel = () => {
    switch (stage) {
      case 'w_projection':
        return 'Stage 1: W-Space Projection'
      case 'pti_tuning':
        return 'Stage 2: Generator Fine-tuning'
      case 'synthesis':
        return 'Synthesis'
      case 'sd_generation':
        return 'Stable Diffusion Generation'
      default:
        return 'Processing'
    }
  }

  // Determine if we should show the toggle (PTI results with progress video)
  const showToggle = status === 'complete' && method === 'upload' && result?.progressVideoUrl

  // Determine what to display
  const getDisplayUrl = () => {
    if (status === 'generating' && previewImage) {
      return previewImage
    }
    if (status === 'complete' && result) {
      if (showToggle && viewMode === 'progress') {
        return result.progressVideoUrl
      }
      return result.imageUrl
    }
    return null
  }

  const displayUrl = getDisplayUrl()
  const isVideo = displayUrl?.endsWith('.mp4') || displayUrl?.endsWith('.webm')

  // Reset video loaded state when URL changes
  useEffect(() => {
    if (isVideo) {
      setVideoLoaded(false)
    }
  }, [displayUrl, isVideo])

  return (
    <div className="h-full flex flex-col">
      {/* Main Preview Area - Auto-sized to fit with bottom bar */}
      <div className="flex-1 flex items-center justify-center p-4 md:p-6 overflow-hidden">
        {status === 'generating' ? (
          /* LIVE PREVIEW MODE - Show optimization progress in real-time */
          <div className="w-full h-full flex items-center justify-center">
            <div className="flex items-center justify-center" style={{ maxWidth: '512px', maxHeight: 'calc(100% - 2rem)' }}>
            {/* Live Preview Image - PROMINENTLY DISPLAYED */}
            {previewImage ? (
              <Card className="overflow-hidden">
                <img
                  src={previewImage}
                  alt="Real-time optimization progress"
                  className="w-auto h-auto max-w-full max-h-full object-contain"
                  style={{ maxWidth: '512px', maxHeight: 'calc(100vh - 400px)' }}
                  onError={(e) => {
                    console.error('[PreviewPanel] Image element error:', e)
                  }}
                />
              </Card>
            ) : (
              /* Show placeholder while waiting for first preview */
              <Card className="overflow-hidden">
                <div className="flex flex-col items-center gap-3 p-20">
                  <Loader2 className="h-12 w-12 animate-spin text-primary" />
                  <p className="text-sm text-muted-foreground">Initializing optimization...</p>
                </div>
              </Card>
            )}
            </div>
          </div>
        ) : displayUrl ? (
          /* COMPLETE/RESULT MODE - Show final result */
          <div className="flex flex-col items-center justify-center gap-3">
            {/* Toggle for PTI results */}
            {showToggle && (
              <div className="flex justify-center">
                <ToggleGroup
                  type="single"
                  value={viewMode}
                  onValueChange={(value) => {
                    if (value) setViewMode(value as 'result' | 'progress')
                  }}
                  className="bg-muted p-1 rounded-lg"
                >
                  <ToggleGroupItem
                    value="result"
                    aria-label="Show result image"
                    className="data-[state=on]:bg-background data-[state=on]:shadow-sm px-4"
                  >
                    <ImageResultIcon className="h-4 w-4 mr-2" />
                    Result
                  </ToggleGroupItem>
                  <ToggleGroupItem
                    value="progress"
                    aria-label="Show progress video"
                    className="data-[state=on]:bg-background data-[state=on]:shadow-sm px-4"
                  >
                    <Film className="h-4 w-4 mr-2" />
                    Progress Video
                  </ToggleGroupItem>
                </ToggleGroup>
              </div>
            )}

            {/* Preview Card */}
            <Card className="overflow-hidden">
              {isVideo ? (
                <div className="relative">
                  {/* Show loading placeholder until video is ready */}
                  {!videoLoaded && (
                    <div className="absolute inset-0 bg-muted flex items-center justify-center min-h-[300px]">
                      <div className="flex flex-col items-center gap-3">
                        <Loader2 className="h-10 w-10 animate-spin text-primary" />
                        <p className="text-sm text-muted-foreground">Loading video...</p>
                      </div>
                    </div>
                  )}
                  <video
                    key={displayUrl}
                    src={displayUrl}
                    controls
                    autoPlay
                    loop
                    className={`w-auto h-auto object-contain ${!videoLoaded ? 'opacity-0' : 'opacity-100'}`}
                    style={{ maxWidth: '512px', maxHeight: 'calc(100vh - 400px)' }}
                    onLoadedData={() => {
                      console.log('[PreviewPanel] Video loaded and ready')
                      setVideoLoaded(true)
                    }}
                    onError={(e) => {
                      console.error('[PreviewPanel] Video loading error:', e)
                      setVideoLoaded(true) // Show video element even on error to display error message
                    }}
                  >
                    Your browser does not support the video tag.
                  </video>
                </div>
              ) : (
                <img
                  src={displayUrl}
                  alt={viewMode === 'progress' ? 'Progress Video' : 'Preview'}
                  className="w-auto h-auto object-contain"
                  style={{ maxWidth: '512px', maxHeight: 'calc(100vh - 400px)' }}
                />
              )}
            </Card>

            {/* View description */}
            {showToggle && (
              <p className="text-xs text-center text-muted-foreground">
                {viewMode === 'result'
                  ? 'Final reconstructed result from PTI projection'
                  : 'Step-by-step progression of the PTI optimization process'}
              </p>
            )}
          </div>
        ) : (
          /* IDLE/ERROR MODE - Show status icon and message */
          <div className="flex flex-col items-center gap-4 text-center max-w-md">
            {statusInfo.icon}
            <div>
              <h3 className="text-lg font-semibold">{statusInfo.title}</h3>
              <p className="text-sm text-muted-foreground mt-1">{statusInfo.description}</p>
            </div>
          </div>
        )}
      </div>

      {/* Bottom Action Bar - Sticky */}
      <div className="border-t bg-background">
        <div className="p-4 md:p-6 space-y-4">
          {/* Progress Bar - Only show during generation */}
          {status === 'generating' && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="font-medium">
                  {progress >= 0.99 ? 'Finishing up...' : getStageLabel()}
                </span>
                <span className="text-muted-foreground">
                  {progress >= 0.99 ? '' : `${formatProgress(progress)} ${totalSteps > 0 ? `(${currentStep}/${totalSteps})` : ''}`}
                </span>
              </div>
              {/* Show indeterminate progress when finishing, normal progress otherwise */}
              {progress >= 0.99 ? (
                <div className="h-2 w-full bg-secondary rounded-full overflow-hidden relative">
                  <div
                    className="h-full bg-primary absolute inset-0"
                    style={{
                      animation: 'indeterminate 1.5s ease-in-out infinite',
                      background: 'linear-gradient(90deg, transparent, hsl(var(--primary)), transparent)',
                      width: '50%',
                    }}
                  />
                  <style>{`
                    @keyframes indeterminate {
                      0% { transform: translateX(-100%); }
                      100% { transform: translateX(300%); }
                    }
                  `}</style>
                </div>
              ) : (
                <Progress value={progress * 100} className="h-2" />
              )}
              {message && progress < 0.99 && (
                <p className="text-sm text-center text-muted-foreground mt-2">{message}</p>
              )}
              {progress >= 0.99 && (
                <p className="text-sm text-center text-muted-foreground mt-2">
                  {method === 'seed' ? 'Generating video and saving results...' :
                   method === 'text' ? 'Finalizing 3D model...' :
                   'Saving results and generating final output...'}
                </p>
              )}
            </div>
          )}

          {/* Success Actions */}
          {status === 'complete' && result && (
            <div className="space-y-3">
              <Alert>
                <CheckCircle2 className="h-4 w-4" />
                <AlertTitle>Success!</AlertTitle>
                <AlertDescription className="text-sm">
                  Your 3D head model has been generated successfully.
                </AlertDescription>
              </Alert>

              <div className="flex gap-2">
                <Button variant="outline" className="flex-1">
                  <Download className="h-4 w-4 mr-2" />
                  Download
                </Button>
                <Button className="flex-1" onClick={handleProceedToCustomization}>
                  Proceed to Customization
                  <ArrowRight className="h-4 w-4 ml-2" />
                </Button>
              </div>
            </div>
          )}

          {/* Error State */}
          {status === 'error' && (
            <Alert variant="destructive">
              <XCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription className="text-sm">{error}</AlertDescription>
            </Alert>
          )}

          {/* Idle State Info */}
          {status === 'idle' && (
            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="p-3 rounded-lg bg-muted/50">
                <div className="text-2xl font-bold text-primary">3</div>
                <div className="text-xs text-muted-foreground mt-1">Generation Methods</div>
              </div>
              <div className="p-3 rounded-lg bg-muted/50">
                <div className="text-2xl font-bold text-primary">360°</div>
                <div className="text-xs text-muted-foreground mt-1">Rotation View</div>
              </div>
              <div className="p-3 rounded-lg bg-muted/50">
                <div className="text-2xl font-bold text-primary">3D</div>
                <div className="text-xs text-muted-foreground mt-1">Head Model</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
