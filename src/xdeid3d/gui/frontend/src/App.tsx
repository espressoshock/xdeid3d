import { AppLayout } from './components/layout/AppLayout'
import { ThemeProvider } from './hooks/useTheme'

function App() {
  return (
    <ThemeProvider>
      <div className="min-h-screen bg-background">
        <AppLayout />
      </div>
    </ThemeProvider>
  )
}

export default App
