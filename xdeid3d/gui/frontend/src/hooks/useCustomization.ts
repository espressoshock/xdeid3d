import { create } from 'zustand'
import { GenerationResult } from '@/types/generation'

export type BackgroundMethod = 'color' | 'image' | 'text'
export type CustomizationStatus = 'idle' | 'applying' | 'complete' | 'error'
export type CameraPoseMode = 'preset' | 'keyframes'
export type InterpolationMode = 'linear' | 'cubic'

export interface BackgroundSettings {
  method: BackgroundMethod
  // For color method
  solidColor?: string
  // For image method
  imageFile?: File
  imageUrl?: string
  imageFit?: 'cover' | 'contain' | 'stretch'
  // For text method
  textPrompt?: string
  negativePrompt?: string
}

export interface CameraKeyframe {
  id: string
  timestamp: number // 0.0 to 1.0 (normalized time)
  yaw: number       // radians
  pitch: number     // radians
  radius: number    // camera distance
}

export interface MeshGenerationState {
  status: 'idle' | 'generating' | 'complete' | 'error'
  progress: number
  message: string
  meshUrl: string | null
  error: string | null
  voxelResolution: number
  isRegenerating: boolean // True when regenerating with different quality (keeps old mesh visible)
}

export interface CameraPoseSettings {
  mode: CameraPoseMode
  preset: string | null
  keyframes: CameraKeyframe[]
  currentTime: number
  duration: number // Total duration in seconds
  interpolation: InterpolationMode
}

export interface PoseVideoState {
  status: 'idle' | 'generating' | 'complete' | 'error'
  progress: number
  message: string
  videoUrl: string | null
  error: string | null
}

export interface PosePreviewSettings {
  frames: number // 30, 60, 120, 240
  fps: number // 24, 30
  quality: 'low' | 'medium' | 'high' | 'best'
}

export interface PosePreviewState {
  viewMode: '3d' | 'rendered'
  status: 'idle' | 'rendering' | 'complete' | 'error'
  progress: number
  message: string
  previewUrl: string | null
  error: string | null
  settings: PosePreviewSettings
}

export interface CustomizationState {
  // Generation result from Stage 1
  baseIdentity: GenerationResult | null
  sessionId: string | null

  // Current customization settings
  backgroundSettings: BackgroundSettings

  // Status tracking
  status: CustomizationStatus
  progress: number
  message: string
  error: string | null

  // Result after customization
  customizedResult: {
    imageUrl: string
    videoUrl?: string
    metadata: Record<string, any>
  } | null

  // === Camera Pose Customization (Stage 2.3) ===
  meshGeneration: MeshGenerationState
  cameraPose: CameraPoseSettings
  poseVideo: PoseVideoState
  posePreview: PosePreviewState

  // Actions
  setBaseIdentity: (result: GenerationResult, sessionId?: string) => void
  setBackgroundMethod: (method: BackgroundMethod) => void
  updateBackgroundSettings: (settings: Partial<BackgroundSettings>) => void
  setStatus: (status: CustomizationStatus) => void
  setProgress: (progress: number, message?: string) => void
  setError: (error: string) => void
  setCustomizedResult: (result: CustomizationState['customizedResult']) => void

  // Camera Pose Actions
  setMeshGenerationStatus: (status: MeshGenerationState['status'], progress?: number, message?: string) => void
  setMeshUrl: (url: string) => void
  setMeshError: (error: string) => void
  setVoxelResolution: (resolution: number) => void
  resetMeshGeneration: () => void
  startMeshRegeneration: () => void
  setCameraPoseMode: (mode: CameraPoseMode) => void
  setPreset: (preset: string) => void
  addKeyframe: (keyframe: Omit<CameraKeyframe, 'id'>) => void
  updateKeyframe: (id: string, updates: Partial<CameraKeyframe>) => void
  removeKeyframe: (id: string) => void
  setCurrentTime: (time: number) => void
  setDuration: (duration: number) => void
  setInterpolation: (interpolation: InterpolationMode) => void
  setPoseVideoStatus: (status: PoseVideoState['status'], progress?: number, message?: string) => void
  setPoseVideoUrl: (url: string) => void
  setPoseVideoError: (error: string) => void

  // Pose Preview Actions
  setPoseViewMode: (mode: '3d' | 'rendered') => void
  setPosePreviewSettings: (settings: Partial<PosePreviewSettings>) => void
  setPosePreviewStatus: (status: PosePreviewState['status'], progress?: number, message?: string) => void
  setPosePreviewUrl: (url: string) => void
  setPosePreviewError: (error: string) => void
  resetPosePreview: () => void

  reset: () => void
}

const initialBackgroundSettings: BackgroundSettings = {
  method: 'color',
  solidColor: '#1a1a1a',
  imageFit: 'cover',
  textPrompt: '',
  negativePrompt: '',
}

const initialMeshGeneration: MeshGenerationState = {
  status: 'idle',
  progress: 0,
  message: '',
  meshUrl: null,
  error: null,
  voxelResolution: 128, // Default quality setting
  isRegenerating: false,
}

const initialCameraPose: CameraPoseSettings = {
  mode: 'preset',
  preset: 'rotate360',
  keyframes: [],
  currentTime: 0,
  duration: 10, // 10 seconds default
  interpolation: 'cubic',
}

const initialPoseVideo: PoseVideoState = {
  status: 'idle',
  progress: 0,
  message: '',
  videoUrl: null,
  error: null,
}

const initialPosePreview: PosePreviewState = {
  viewMode: '3d',
  status: 'idle',
  progress: 0,
  message: '',
  previewUrl: null,
  error: null,
  settings: {
    frames: 60,
    fps: 24,
    quality: 'medium',
  },
}

const initialState = {
  baseIdentity: null,
  sessionId: null,
  backgroundSettings: initialBackgroundSettings,
  status: 'idle' as CustomizationStatus,
  progress: 0,
  message: '',
  error: null,
  customizedResult: null,
  meshGeneration: initialMeshGeneration,
  cameraPose: initialCameraPose,
  poseVideo: initialPoseVideo,
  posePreview: initialPosePreview,
}

export const useCustomization = create<CustomizationState>((set) => ({
  ...initialState,

  setBaseIdentity: (result, sessionId) =>
    set({
      baseIdentity: result,
      sessionId: sessionId || `custom-${Date.now()}`,
      status: 'idle',
      progress: 0,
      message: '',
      error: null,
      customizedResult: null, // Clear previous customization when new identity is set
    }),

  setBackgroundMethod: (method) =>
    set((state) => ({
      backgroundSettings: {
        ...state.backgroundSettings,
        method,
      },
      error: null,
    })),

  updateBackgroundSettings: (settings) =>
    set((state) => ({
      backgroundSettings: {
        ...state.backgroundSettings,
        ...settings,
      },
      error: null, // Clear errors when user changes settings
    })),

  setStatus: (status) => set({ status }),

  setProgress: (progress, message = '') =>
    set({ progress, message, status: 'applying' }),

  setError: (error) =>
    set({ status: 'error', error }),

  setCustomizedResult: (result) =>
    set({ customizedResult: result, status: 'complete', progress: 1 }),

  // Camera Pose Actions
  setMeshGenerationStatus: (status, progress = 0, message = '') =>
    set((state) => ({
      meshGeneration: {
        ...state.meshGeneration,
        status,
        progress,
        message,
        error: status === 'error' ? state.meshGeneration.error : null,
      },
    })),

  setMeshUrl: (url) =>
    set((state) => ({
      meshGeneration: {
        ...state.meshGeneration,
        meshUrl: url,
        status: 'complete',
        progress: 1,
        message: 'Mesh generated successfully',
        isRegenerating: false, // Clear regenerating flag
      },
    })),

  setMeshError: (error) =>
    set((state) => ({
      meshGeneration: {
        ...state.meshGeneration,
        status: 'error',
        error,
        isRegenerating: false, // Clear regenerating flag on error
      },
    })),

  setVoxelResolution: (resolution) =>
    set((state) => ({
      meshGeneration: {
        ...state.meshGeneration,
        voxelResolution: resolution,
      },
    })),

  resetMeshGeneration: () =>
    set((state) => ({
      meshGeneration: {
        ...initialMeshGeneration,
        voxelResolution: state.meshGeneration.voxelResolution, // Preserve voxel resolution
      },
    })),

  startMeshRegeneration: () =>
    set((state) => ({
      meshGeneration: {
        ...state.meshGeneration,
        status: 'generating',
        progress: 0,
        message: 'Regenerating mesh...',
        error: null,
        isRegenerating: true, // Set regenerating flag (preserves meshUrl)
      },
    })),

  setCameraPoseMode: (mode) =>
    set((state) => ({
      cameraPose: {
        ...state.cameraPose,
        mode,
      },
    })),

  setPreset: (preset) =>
    set((state) => ({
      cameraPose: {
        ...state.cameraPose,
        preset,
        mode: 'preset',
      },
    })),

  addKeyframe: (keyframe) =>
    set((state) => ({
      cameraPose: {
        ...state.cameraPose,
        keyframes: [
          ...state.cameraPose.keyframes,
          {
            ...keyframe,
            id: `kf-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          },
        ].sort((a, b) => a.timestamp - b.timestamp), // Keep sorted by timestamp
      },
    })),

  updateKeyframe: (id, updates) =>
    set((state) => ({
      cameraPose: {
        ...state.cameraPose,
        keyframes: state.cameraPose.keyframes
          .map((kf) => (kf.id === id ? { ...kf, ...updates } : kf))
          .sort((a, b) => a.timestamp - b.timestamp), // Re-sort after update
      },
    })),

  removeKeyframe: (id) =>
    set((state) => ({
      cameraPose: {
        ...state.cameraPose,
        keyframes: state.cameraPose.keyframes.filter((kf) => kf.id !== id),
      },
    })),

  setCurrentTime: (time) =>
    set((state) => ({
      cameraPose: {
        ...state.cameraPose,
        currentTime: Math.max(0, Math.min(1, time)), // Clamp to [0, 1]
      },
    })),

  setDuration: (duration) =>
    set((state) => ({
      cameraPose: {
        ...state.cameraPose,
        duration: Math.max(1, duration), // Minimum 1 second
      },
    })),

  setInterpolation: (interpolation) =>
    set((state) => ({
      cameraPose: {
        ...state.cameraPose,
        interpolation,
      },
    })),

  setPoseVideoStatus: (status, progress = 0, message = '') =>
    set((state) => ({
      poseVideo: {
        ...state.poseVideo,
        status,
        progress,
        message,
        error: status === 'error' ? state.poseVideo.error : null,
      },
    })),

  setPoseVideoUrl: (url) =>
    set((state) => ({
      poseVideo: {
        ...state.poseVideo,
        videoUrl: url,
        status: 'complete',
        progress: 1,
        message: 'Pose video generated successfully',
      },
    })),

  setPoseVideoError: (error) =>
    set((state) => ({
      poseVideo: {
        ...state.poseVideo,
        status: 'error',
        error,
      },
    })),

  // Pose Preview Actions
  setPoseViewMode: (mode) =>
    set((state) => ({
      posePreview: {
        ...state.posePreview,
        viewMode: mode,
      },
    })),

  setPosePreviewSettings: (settings) =>
    set((state) => ({
      posePreview: {
        ...state.posePreview,
        settings: {
          ...state.posePreview.settings,
          ...settings,
        },
        // Reset preview when settings change
        status: 'idle',
        previewUrl: null,
        error: null,
      },
    })),

  setPosePreviewStatus: (status, progress = 0, message = '') =>
    set((state) => ({
      posePreview: {
        ...state.posePreview,
        status,
        progress,
        message,
        error: status === 'error' ? state.posePreview.error : null,
      },
    })),

  setPosePreviewUrl: (url) =>
    set((state) => ({
      posePreview: {
        ...state.posePreview,
        previewUrl: url,
        status: 'complete',
        progress: 1,
        message: 'Preview rendered successfully',
        error: null,
      },
    })),

  setPosePreviewError: (error) =>
    set((state) => ({
      posePreview: {
        ...state.posePreview,
        status: 'error',
        error,
        progress: 0,
      },
    })),

  resetPosePreview: () =>
    set((state) => ({
      posePreview: {
        ...initialPosePreview,
        settings: state.posePreview.settings, // Preserve settings
      },
    })),

  reset: () => set(initialState),
}))
