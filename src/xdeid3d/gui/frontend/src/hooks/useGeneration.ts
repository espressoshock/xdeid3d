import { create } from 'zustand'
import { GenerationState, GenerationMethod, WebSocketMessage } from '@/types/generation'

interface GenerationStore extends GenerationState {
  setMethod: (method: GenerationMethod) => void
  handleWebSocketMessage: (message: WebSocketMessage) => void
  reset: () => void
}

const initialState: GenerationState = {
  method: 'seed',
  status: 'idle',
  progress: 0,
  stage: null,
  currentStep: 0,
  totalSteps: 0,
  message: '',
  previewImage: null,
  result: null,
  error: null,
}

export const useGeneration = create<GenerationStore>((set) => ({
  ...initialState,

  setMethod: (method) => set({ method, status: 'idle', error: null }),

  handleWebSocketMessage: (message) => {
    console.log('[useGeneration] Handling message type:', message.type)

    switch (message.type) {
      case 'connected':
        // Connection established successfully
        console.log('[WS] Connected:', message.session_id)
        break

      case 'heartbeat':
      case 'pong':
        // Keepalive messages, ignore silently
        console.log('[WS] Received keepalive:', message.type)
        break

      case 'progress':
        // Log preview image updates for debugging
        if (message.preview_image) {
          console.log('[useGeneration] Preview image received:', message.preview_image)
        }

        set((state) => ({
          status: 'generating',
          progress: message.progress || 0,
          stage: message.stage || null,
          currentStep: message.step || 0,
          totalSteps: message.total_steps || 0,
          message: message.message || '',
          // Only update previewImage if the message contains one, otherwise keep existing
          previewImage: message.preview_image !== undefined ? message.preview_image : state.previewImage,
          error: null,
        }))
        break

      case 'complete':
        set({
          status: 'complete',
          progress: 1,
          message: 'Generation complete!',
          result: message.result || null,
          error: null,
        })
        break

      case 'error':
        set({
          status: 'error',
          error: message.error || 'An unknown error occurred',
          message: message.error || '',
        })
        break
    }
  },

  reset: () => set(initialState),
}))
