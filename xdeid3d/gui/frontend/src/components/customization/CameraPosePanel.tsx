import { useEffect, useState } from 'react'
import { useCustomization } from '@/hooks/useCustomization'
import { useWebSocket } from '@/hooks/useWebSocket'
import { api } from '@/lib/api'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Separator } from '@/components/ui/separator'
import { AlertCircle, Loader2, Check } from 'lucide-react'
import { MeshGenerationStatus } from './MeshGenerationStatus'
import { ThreeDViewer } from './ThreeDViewer'
import { CameraControls } from './CameraControls'
import { MeshSettings } from './MeshSettings'
import { PoseViewToggle } from './PoseViewToggle'
import { PosePreviewRenderer } from './PosePreviewRenderer'
import { motion, AnimatePresence } from 'framer-motion'

export function CameraPosePanel() {
  const {
    baseIdentity,
    sessionId,
    meshGeneration,
    posePreview,
    cameraPose,
    setMeshGenerationStatus,
    setMeshUrl,
    setMeshError,
    setCurrentTime,
    addKeyframe,
    setPosePreviewStatus,
  } = useCustomization()

  // Visual feedback for keyframe addition
  const [keyframeAdded, setKeyframeAdded] = useState(false)

  // Helper function to add keyframe with smart timestamp selection
  const handleAddKeyframe = (pose: { yaw: number; pitch: number; radius: number }) => {
    const { currentTime, keyframes } = cameraPose

    // Find a good timestamp to use
    let targetTime = currentTime

    // Check if there's already a keyframe at this exact time
    const existingAtTime = keyframes.find(kf => Math.abs(kf.timestamp - targetTime) < 0.01)

    if (existingAtTime) {
      // Auto-increment to find next available slot
      // Space keyframes evenly by 0.1 (10% of timeline)
      const increment = 0.1
      let attempts = 0
      while (attempts < 10) {
        targetTime = currentTime + (increment * (attempts + 1))
        if (targetTime > 1.0) {
          // Wrap around if we exceed timeline
          targetTime = currentTime - (increment * (attempts + 1))
        }

        const conflictExists = keyframes.find(kf => Math.abs(kf.timestamp - targetTime) < 0.01)
        if (!conflictExists && targetTime >= 0 && targetTime <= 1.0) {
          break
        }
        attempts++
      }

      // Clamp to valid range
      targetTime = Math.max(0, Math.min(1, targetTime))
    }

    // Add the keyframe
    addKeyframe({
      timestamp: targetTime,
      yaw: pose.yaw,
      pitch: pose.pitch,
      radius: pose.radius,
    })

    // Move playhead to the new keyframe position
    setCurrentTime(targetTime)

    // Show visual feedback
    setKeyframeAdded(true)
    setTimeout(() => setKeyframeAdded(false), 2000)

    console.log('[CameraPosePanel] Keyframe added at timestamp:', targetTime, 'pose:', pose)
  }

  // WebSocket connection for real-time mesh generation progress
  const wsUrl = sessionId ? `ws://localhost:8000/ws/generation/${sessionId}` : null

  useWebSocket(wsUrl, {
    onMessage: (message) => {
      console.log('[CameraPosePanel] WebSocket message:', message)

      // Handle mesh generation progress
      if (message.type === 'mesh_progress') {
        const { progress, message: statusMessage } = message as any
        console.log('[CameraPosePanel] Mesh progress:', progress, statusMessage)
        setMeshGenerationStatus('generating', progress, statusMessage || 'Extracting 3D mesh...')
      }

      // Handle pose preview rendering progress
      if (message.type === 'pose_preview_progress') {
        const { progress, message: statusMessage } = message as any
        console.log('[CameraPosePanel] Pose preview progress:', progress, statusMessage)
        setPosePreviewStatus('rendering', progress, statusMessage || 'Rendering preview...')
      }
    },
  })

  // Auto-trigger mesh generation on mount
  useEffect(() => {
    if (!baseIdentity || !sessionId) return

    // Only generate if mesh hasn't been generated yet
    if (meshGeneration.status === 'idle' && !meshGeneration.meshUrl) {
      console.log('[CameraPosePanel] Auto-triggering mesh generation')
      triggerMeshGeneration()
    }
  }, [baseIdentity, sessionId, meshGeneration.status, meshGeneration.meshUrl])

  const triggerMeshGeneration = async () => {
    if (!baseIdentity || !sessionId) {
      console.error('[CameraPosePanel] Missing baseIdentity or sessionId')
      return
    }

    try {
      setMeshGenerationStatus('generating', 0, 'Initializing mesh extraction...')

      console.log('[CameraPosePanel] Calling mesh generation API with:', {
        sessionId,
        latentCodePath: baseIdentity.latentCodePath,
        generatorPath: baseIdentity.generatorPath,
        seed: baseIdentity.metadata?.seed,
        truncation: baseIdentity.metadata?.params?.truncation || 0.65,
      })

      // Call mesh generation API
      // Progress updates will come via WebSocket (mesh_progress messages)
      const response = await api.generateMesh({
        session_id: sessionId,
        latent_code_path: baseIdentity.latentCodePath,
        generator_path: baseIdentity.generatorPath,
        seed: baseIdentity.metadata?.seed,
        truncation: baseIdentity.metadata?.params?.truncation || baseIdentity.metadata?.truncation || 0.65,
        voxel_res: meshGeneration.voxelResolution, // Use voxel resolution from state
      })

      if (response.success && response.mesh_url) {
        console.log('[CameraPosePanel] Mesh generated successfully:', response.mesh_url)
        setMeshUrl(response.mesh_url)
      } else {
        throw new Error(response.error || response.message || 'Mesh generation failed')
      }
    } catch (error) {
      console.error('[CameraPosePanel] Mesh generation failed:', error)
      setMeshError(error instanceof Error ? error.message : 'Failed to generate mesh')
    }
  }

  // Show warning if no base identity
  if (!baseIdentity || !sessionId) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <Alert className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>No Identity Generated</AlertTitle>
          <AlertDescription>
            Please complete the generation stage first to create a base identity before customizing camera poses.
          </AlertDescription>
        </Alert>
      </div>
    )
  }

  // Show mesh generation status (only for initial generation, not regeneration)
  // During regeneration, we show the main UI with progress in MeshSettings
  const showFullscreenStatus =
    (meshGeneration.status === 'generating' && !meshGeneration.isRegenerating) ||
    (meshGeneration.status === 'error' && !meshGeneration.meshUrl) // Only show fullscreen error if no mesh exists yet

  if (showFullscreenStatus) {
    return (
      <MeshGenerationStatus
        status={meshGeneration.status}
        progress={meshGeneration.progress}
        message={meshGeneration.message}
        error={meshGeneration.error}
        onRetry={triggerMeshGeneration}
      />
    )
  }

  // Main UI (when mesh is ready or will be shown once generated)
  return (
    <div className="h-full flex">
      {/* Left Panel - Camera Controls */}
      <div className="w-[420px] border-r flex flex-col">
        <div className="p-6 flex-1 overflow-y-auto">
          <div className="mb-4">
            <h2 className="text-lg font-semibold mb-1">Customize Camera Pose</h2>
            <p className="text-sm text-muted-foreground">
              Define custom camera trajectories using presets or manual keyframes
            </p>
          </div>

          <Separator className="my-4" />

          {/* Mesh Quality Settings */}
          <MeshSettings />

          <Separator className="my-4" />

          {/* Camera Controls */}
          <CameraControls />
        </div>

        {/* Status Footer */}
        <div className="px-6 py-3 border-t">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <div className={`h-2 w-2 rounded-full ${meshGeneration.meshUrl ? 'bg-green-500' : 'bg-yellow-500'}`} />
            {meshGeneration.meshUrl ? 'Mesh ready' : 'Waiting for mesh'}
          </div>
        </div>
      </div>

      {/* Right Panel - 3D Viewer / Preview Renderer */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {/* View Mode Toggle */}
        <PoseViewToggle />

        {/* Viewer Area */}
        <div className="flex-1 overflow-hidden">
          {posePreview.viewMode === '3d' ? (
            // 3D Model View
            meshGeneration.meshUrl ? (
              <ThreeDViewer
                meshUrl={meshGeneration.meshUrl}
                onCaptureView={handleAddKeyframe}
              />
            ) : (
              <div className="h-full flex items-center justify-center bg-[#1A1E37] text-center text-muted-foreground">
                <div>
                  <Loader2 className="h-8 w-8 animate-spin mx-auto mb-3" />
                  <p>Preparing 3D viewer...</p>
                </div>
              </div>
            )
          ) : (
            // Rendered Pose Preview
            <PosePreviewRenderer />
          )}
        </div>

        {/* Visual Feedback - Keyframe Added */}
        <AnimatePresence>
          {keyframeAdded && (
            <motion.div
              initial={{ opacity: 0, y: 20, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -20, scale: 0.9 }}
              transition={{ duration: 0.2 }}
              className="absolute bottom-6 left-1/2 -translate-x-1/2 z-50"
            >
              <div className="bg-primary text-primary-foreground px-4 py-3 rounded-lg shadow-lg flex items-center gap-2">
                <Check className="h-4 w-4" />
                <span className="text-sm font-medium">Keyframe added!</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
