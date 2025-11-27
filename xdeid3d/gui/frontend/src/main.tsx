import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

// Note: StrictMode disabled to prevent WebSocket reconnection issues in development
// StrictMode causes double-mounting which closes/reopens WebSocket connections
// Re-enable for production builds or when using proper cleanup
ReactDOM.createRoot(document.getElementById('root')!).render(
  <App />
)
