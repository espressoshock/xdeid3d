import { useEffect, useRef, useState, useCallback } from 'react'
import { WebSocketMessage } from '@/types/generation'

export interface UseWebSocketOptions {
  onMessage?: (message: WebSocketMessage) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
}

export function useWebSocket(url: string | null, options: UseWebSocketOptions = {}) {
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const optionsRef = useRef(options)
  const isConnectingRef = useRef(false)

  // Update options ref without triggering reconnection
  useEffect(() => {
    optionsRef.current = options
  }, [options])

  const connect = useCallback(() => {
    // Prevent multiple simultaneous connection attempts
    if (!url) {
      console.log('[WebSocket] No URL provided, skipping connection')
      return
    }

    const currentState = wsRef.current?.readyState
    if (currentState === WebSocket.OPEN || currentState === WebSocket.CONNECTING) {
      console.log('[WebSocket] Already connected or connecting, skipping')
      return
    }

    if (isConnectingRef.current) {
      console.log('[WebSocket] Connection attempt already in progress, skipping')
      return
    }

    isConnectingRef.current = true
    console.log('[WebSocket] Starting new connection...')

    try {
      const ws = new WebSocket(url)

      ws.onopen = () => {
        console.log('[WebSocket] Connected successfully')
        isConnectingRef.current = false
        setIsConnected(true)
        setError(null)
        reconnectAttemptsRef.current = 0

        // Send initial ping to establish connection
        try {
          ws.send(JSON.stringify({ type: 'ping' }))
          console.log('[WebSocket] Sent initial ping')
        } catch (e) {
          console.error('[WebSocket] Failed to send ping:', e)
        }

        optionsRef.current.onConnect?.()
      }

      ws.onmessage = (event) => {
        try {
          console.log('[WebSocket] Raw message received:', event.data)
          const message: WebSocketMessage = JSON.parse(event.data)
          console.log('[WebSocket] Parsed message:', message)

          if (optionsRef.current.onMessage) {
            console.log('[WebSocket] Calling onMessage handler')
            optionsRef.current.onMessage(message)
            console.log('[WebSocket] onMessage handler completed')
          } else {
            console.warn('[WebSocket] No onMessage handler defined')
          }
        } catch (err) {
          console.error('[WebSocket] Failed to parse message:', err, event.data)
        }
      }

      ws.onerror = (event) => {
        console.error('[WebSocket] Error:', event)
        setError('WebSocket connection error')
        optionsRef.current.onError?.(event)
      }

      ws.onclose = (event) => {
        console.log('[WebSocket] CLOSE EVENT - Code:', event.code, 'Reason:', event.reason || 'none', 'wasClean:', event.wasClean)
        console.log('[WebSocket] Current readyState:', ws?.readyState)
        isConnectingRef.current = false
        setIsConnected(false)

        if (optionsRef.current.onDisconnect) {
          console.log('[WebSocket] Calling onDisconnect handler')
          optionsRef.current.onDisconnect()
        }

        // Only reconnect on abnormal closure and if URL is still valid
        // Normal closure (1000) means intentional disconnect
        // Code 1005 is also considered abnormal (no status received)
        if (event.code !== 1000 && url && reconnectAttemptsRef.current < 3) {
          const delay = Math.min(2000 * Math.pow(2, reconnectAttemptsRef.current), 15000)
          console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current + 1}/3)`)
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++
            connect()
          }, delay)
        } else if (reconnectAttemptsRef.current >= 3) {
          console.log('[WebSocket] Max reconnection attempts reached')
          setError('Connection lost. Please refresh the page.')
        } else if (event.code === 1000) {
          console.log('[WebSocket] Normal closure, not reconnecting')
        }
      }

      wsRef.current = ws
    } catch (err) {
      console.error('[WebSocket] Connection failed:', err)
      isConnectingRef.current = false
      setError('Failed to connect to server')
    }
  }, [url])

  const disconnect = useCallback(() => {
    console.log('[WebSocket] Disconnecting...')
    isConnectingRef.current = false
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsConnected(false)
  }, [])

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    } else {
      console.warn('[WebSocket] Cannot send message, not connected')
    }
  }, [])

  useEffect(() => {
    if (url) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [url, connect, disconnect])

  return {
    isConnected,
    error,
    sendMessage,
    disconnect,
  }
}
