import { useEffect, useRef, useState } from 'react'
import { useCustomization } from '@/hooks/useCustomization'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Loader2, AlertCircle, RotateCcw, Box, Play, Pause, RotateCw, Video } from 'lucide-react'

export function PosePreviewRenderer() {
  const { posePreview, cameraPose, setPoseViewMode, resetPosePreview } = useCustomization()
  const { status, progress, message, previewUrl, error, settings } = posePreview

  const videoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isLooping, setIsLooping] = useState(true)
  const [videoKey, setVideoKey] = useState(0)

  // Auto-play when video loads or changes
  useEffect(() => {
    if (videoRef.current && previewUrl) {
      console.log('[PosePreviewRenderer] New preview URL, reloading video:', previewUrl)

      // Force reload by updating key (remounts video element)
      setVideoKey(prev => prev + 1)

      // Small delay to ensure video element is ready
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.load()
          videoRef.current.play().catch(err => {
            console.log('[PosePreviewRenderer] Autoplay prevented:', err)
          })
          setIsPlaying(true)
        }
      }, 100)
    }
  }, [previewUrl])

  // Handle play/pause
  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
      } else {
        videoRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const toggleLoop = () => {
    if (videoRef.current) {
      videoRef.current.loop = !isLooping
      setIsLooping(!isLooping)
    }
  }

  const handleRetry = () => {
    resetPosePreview()
    // Preview will be re-triggered by CameraControls auto-update logic
  }

  const handleSwitchTo3D = () => {
    setPoseViewMode('3d')
  }

  // Rendering state
  if (status === 'rendering') {
    return (
      <div className="h-full flex items-center justify-center bg-[#1A1E37] p-6">
        <div className="max-w-md w-full space-y-6 text-center">
          <Loader2 className="h-12 w-12 text-primary animate-spin mx-auto" />

          <div className="space-y-2">
            <h3 className="text-lg font-semibold">Rendering Preview</h3>
            <p className="text-sm text-muted-foreground">
              {message || 'Generating frames...'}
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Progress</span>
              <span className="font-mono">{Math.round(progress * 100)}%</span>
            </div>
            <Progress value={progress * 100} className="h-2" />
          </div>

          <div className="grid grid-cols-3 gap-2 text-xs text-muted-foreground">
            <div>
              <div className="font-medium text-foreground">{settings.frames}</div>
              <div>Frames</div>
            </div>
            <div>
              <div className="font-medium text-foreground">{settings.fps} fps</div>
              <div>Frame Rate</div>
            </div>
            <div>
              <div className="font-medium text-foreground capitalize">{settings.quality}</div>
              <div>Quality</div>
            </div>
          </div>

          <p className="text-xs text-muted-foreground">
            Estimated time: {settings.quality === 'low' ? '~30s' : settings.quality === 'medium' ? '~1-2min' : '~3-5min'}
          </p>
        </div>
      </div>
    )
  }

  // Error state
  if (status === 'error') {
    return (
      <div className="h-full flex items-center justify-center bg-[#1A1E37] p-6">
        <div className="max-w-md w-full space-y-4">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Rendering Failed</AlertTitle>
            <AlertDescription className="text-sm">
              {error || 'An error occurred while rendering the preview'}
            </AlertDescription>
          </Alert>

          <div className="flex gap-2">
            <Button
              onClick={handleRetry}
              className="flex-1"
              variant="outline"
            >
              <RotateCcw className="h-4 w-4 mr-2" />
              Retry
            </Button>
            <Button
              onClick={handleSwitchTo3D}
              className="flex-1"
            >
              <Box className="h-4 w-4 mr-2" />
              Switch to 3D Model
            </Button>
          </div>

          <p className="text-xs text-center text-muted-foreground">
            Check that your identity is properly generated and the backend server is running
          </p>
        </div>
      </div>
    )
  }

  // Complete state with video
  if (status === 'complete' && previewUrl) {
    const duration = (settings.frames / settings.fps).toFixed(2)

    return (
      <div className="h-full flex flex-col items-center justify-center bg-[#1A1E37] p-6 overflow-auto">
        <div className="w-full max-w-[512px] space-y-4">
          {/* Video Player */}
          <Card className="overflow-hidden bg-black">
            <video
              key={videoKey}
              ref={videoRef}
              src={`http://localhost:8000${previewUrl}`}
              className="w-full h-auto"
              loop={isLooping}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              controls
            />
          </Card>

          {/* Video Controls */}
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <Button
                onClick={togglePlayPause}
                size="sm"
                variant="outline"
              >
                {isPlaying ? (
                  <>
                    <Pause className="h-3.5 w-3.5 mr-1" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="h-3.5 w-3.5 mr-1" />
                    Play
                  </>
                )}
              </Button>

              <Button
                onClick={toggleLoop}
                size="sm"
                variant={isLooping ? "default" : "outline"}
              >
                <RotateCw className="h-3.5 w-3.5 mr-1" />
                Loop
              </Button>
            </div>

            <div className="flex items-center gap-4 text-xs text-muted-foreground">
              <div>
                <span className="font-medium text-foreground">{settings.frames}</span> frames
              </div>
              <div>
                <span className="font-medium text-foreground">{settings.fps}</span> fps
              </div>
              <div>
                <span className="font-medium text-foreground">{duration}s</span> duration
              </div>
              <div>
                <span className="font-medium text-foreground capitalize">{settings.quality}</span> quality
              </div>
            </div>
          </div>

          {/* Success Message */}
          <Alert>
            <AlertDescription className="text-sm">
              Preview rendered successfully! This is a quick preview - adjust settings for different quality levels.
            </AlertDescription>
          </Alert>
        </div>
      </div>
    )
  }

  // Idle state (no preview yet)
  // Check if user has enough keyframes
  const hasInsufficientKeyframes = cameraPose.mode === 'keyframes' && cameraPose.keyframes.length < 2

  return (
    <div className="h-full flex items-center justify-center bg-[#1A1E37] p-6">
      <div className="text-center max-w-md space-y-4">
        {hasInsufficientKeyframes ? (
          <>
            <AlertCircle className="h-12 w-12 mx-auto text-yellow-500 opacity-75" />
            <div>
              <h3 className="text-lg font-semibold mb-2">Need More Keyframes</h3>
              <p className="text-sm text-muted-foreground mb-4">
                You need at least 2 keyframes to generate a preview animation.
              </p>
              <div className="text-xs text-muted-foreground text-left bg-background/50 p-3 rounded-md">
                <p className="font-medium mb-2">To add keyframes:</p>
                <ol className="list-decimal list-inside space-y-1">
                  <li>Switch to 3D view mode (toggle above)</li>
                  <li>Rotate the 3D model to your desired angle</li>
                  <li>Click "Add Keyframe" button</li>
                  <li>Repeat for different camera angles</li>
                </ol>
              </div>
              <p className="text-xs text-muted-foreground mt-3">
                Currently have: <span className="font-medium text-yellow-500">{cameraPose.keyframes.length}</span> keyframe{cameraPose.keyframes.length !== 1 ? 's' : ''}
              </p>
            </div>
            <Button onClick={() => setPoseViewMode('3d')} className="mt-4">
              <Box className="h-4 w-4 mr-2" />
              Switch to 3D View
            </Button>
          </>
        ) : (
          <>
            <Video className="h-12 w-12 mx-auto mb-3 text-muted-foreground opacity-50" />
            <p className="text-sm text-muted-foreground">
              {cameraPose.mode === 'preset'
                ? 'Select a camera preset to generate a preview'
                : 'Preview will render automatically when you add keyframes'
              }
            </p>
            <p className="text-xs text-muted-foreground opacity-75">
              Adjust settings below to customize quality and duration
            </p>
          </>
        )}
      </div>
    </div>
  )
}
