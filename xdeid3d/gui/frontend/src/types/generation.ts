export type GenerationMethod = 'seed' | 'upload' | 'text'

export type GenerationStatus = 'idle' | 'generating' | 'complete' | 'error'

export type GenerationStage =
  | 'w_projection'
  | 'pti_tuning'
  | 'synthesis'
  | 'sd_generation'

export interface GenerationState {
  method: GenerationMethod
  status: GenerationStatus
  progress: number
  stage: GenerationStage | null
  currentStep: number
  totalSteps: number
  message: string
  previewImage: string | null
  result: GenerationResult | null
  error: string | null
}

export interface GenerationResult {
  imageUrl: string
  depthUrl?: string
  progressVideoUrl?: string
  latentCodePath: string
  generatorPath?: string
  metadata: {
    seed?: number
    method: GenerationMethod
    params: Record<string, any>
  }
}

export interface SeedGenerationParams {
  seed: number
  truncation: number
  nrr: number
  sampleMult: number
}

export interface TextGenerationParams {
  prompt: string
  negativePrompt: string
  steps: number
  guidanceScale: number
}

export interface WebSocketMessage {
  type: 'connected' | 'heartbeat' | 'pong' | 'progress' | 'complete' | 'error'
  session_id?: string
  stage?: GenerationStage
  progress?: number
  step?: number
  total_steps?: number
  message?: string
  preview_image?: string
  result?: GenerationResult
  error?: string
  code?: string
}

export interface QualityPreset {
  name: string
  label: string
  description: string
  params: {
    truncation: number
    nrr: number
    sampleMult: number
  }
  ptiSteps?: {
    wSteps: number
    ptiSteps: number
  }
}

export const QUALITY_PRESETS: QualityPreset[] = [
  {
    name: 'fast',
    label: 'Fast',
    description: 'Quick generation, lower quality',
    params: {
      truncation: 0.7,
      nrr: 64,
      sampleMult: 1.0,
    },
    ptiSteps: {
      wSteps: 300,
      ptiSteps: 200,
    },
  },
  {
    name: 'balanced',
    label: 'Balanced',
    description: 'Good balance of speed and quality',
    params: {
      truncation: 0.65,
      nrr: 128,
      sampleMult: 1.5,
    },
    ptiSteps: {
      wSteps: 500,
      ptiSteps: 350,
    },
  },
  {
    name: 'high',
    label: 'High Quality',
    description: 'Best quality, slower generation',
    params: {
      truncation: 0.6,
      nrr: 512,
      sampleMult: 2.5,
    },
    ptiSteps: {
      wSteps: 800,
      ptiSteps: 500,
    },
  },
]
