import { useState, useEffect } from 'react'
import { useCustomization } from '@/hooks/useCustomization'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Separator } from '@/components/ui/separator'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import {
  Loader2,
  CheckCircle2,
  XCircle,
  Download,
  ArrowRight,
  Image as ImageIcon,
  Split,
  Wand2,
} from 'lucide-react'

export function CustomizationPreview() {
  const {
    baseIdentity,
    status,
    progress,
    message,
    error,
    customizedResult,
  } = useCustomization()

  const [viewMode, setViewMode] = useState<'original' | 'customized' | 'split'>('original')
  const [cacheBuster, setCacheBuster] = useState(Date.now())

  // Auto-switch to customized view when a new result is available
  // Also update cache buster to force image reload
  useEffect(() => {
    if (customizedResult) {
      const newBuster = Date.now()
      setViewMode('customized')
      setCacheBuster(newBuster) // Force image reload by updating timestamp
      console.log('[CustomizationPreview] New result - cache buster:', newBuster, 'URL:', customizedResult.imageUrl)
    }
  }, [customizedResult]) // Trigger on ANY change to customizedResult object

  // Add cache-busting parameter to force browser to reload updated images
  const addCacheBuster = (url: string) => {
    if (!url) return url
    const separator = url.includes('?') ? '&' : '?'
    return `${url}${separator}t=${cacheBuster}`
  }

  const getStatusInfo = () => {
    switch (status) {
      case 'idle':
        return {
          title: 'Configure Background',
          description: 'Adjust settings on the left panel, then apply',
          icon: <ImageIcon className="h-12 w-12 text-muted-foreground" />,
        }
      case 'applying':
        return {
          title: 'Applying Background...',
          description: message || 'Processing your customization',
          icon: <Loader2 className="h-12 w-12 text-primary animate-spin" />,
        }
      case 'complete':
        return {
          title: 'Customization Complete',
          description: 'Background has been successfully applied',
          icon: <CheckCircle2 className="h-12 w-12 text-green-500" />,
        }
      case 'error':
        return {
          title: 'Customization Failed',
          description: error || 'An error occurred',
          icon: <XCircle className="h-12 w-12 text-destructive" />,
        }
    }
  }

  const statusInfo = getStatusInfo()

  return (
    <div className="h-full flex flex-col p-6 overflow-auto">
      {/* Main Preview Area */}
      <div className="flex-1 flex items-center justify-center">
        {(status === 'idle' || status === 'complete') && baseIdentity && (
          <div className="max-w-[512px] w-full space-y-4">
            {/* View Mode Toggle */}
            {customizedResult && (
              <div className="flex justify-center">
                <ToggleGroup
                  type="single"
                  value={viewMode}
                  onValueChange={(value) => {
                    if (value) setViewMode(value as typeof viewMode)
                  }}
                  className="bg-muted p-1 rounded-lg"
                >
                  <ToggleGroupItem
                    value="original"
                    className="data-[state=on]:bg-background data-[state=on]:shadow-sm px-4"
                  >
                    <ImageIcon className="h-4 w-4 mr-2" />
                    Original
                  </ToggleGroupItem>
                  <ToggleGroupItem
                    value="customized"
                    className="data-[state=on]:bg-background data-[state=on]:shadow-sm px-4"
                  >
                    <Wand2 className="h-4 w-4 mr-2" />
                    Customized
                  </ToggleGroupItem>
                  <ToggleGroupItem
                    value="split"
                    className="data-[state=on]:bg-background data-[state=on]:shadow-sm px-4"
                  >
                    <Split className="h-4 w-4 mr-2" />
                    Split View
                  </ToggleGroupItem>
                </ToggleGroup>
              </div>
            )}

            {/* Preview Card */}
            <Card className="overflow-hidden">
              {viewMode === 'split' && customizedResult ? (
                <div className="grid grid-cols-2">
                  <div className="border-r">
                    <img
                      src={baseIdentity.imageUrl}
                      alt="Original"
                      className="w-full h-auto"
                    />
                    <div className="p-2 text-center text-xs text-muted-foreground bg-muted/50">
                      Original
                    </div>
                  </div>
                  <div>
                    <img
                      src={addCacheBuster(customizedResult.imageUrl)}
                      alt="Customized"
                      className="w-full h-auto"
                    />
                    <div className="p-2 text-center text-xs text-muted-foreground bg-muted/50">
                      Customized
                    </div>
                  </div>
                </div>
              ) : (
                <img
                  src={
                    viewMode === 'customized' && customizedResult
                      ? addCacheBuster(customizedResult.imageUrl)
                      : baseIdentity.imageUrl
                  }
                  alt={viewMode === 'customized' ? 'Customized' : 'Original'}
                  className="w-full h-auto"
                />
              )}
            </Card>

            <p className="text-xs text-center text-muted-foreground">
              {viewMode === 'original'
                ? 'Original identity from generation stage'
                : viewMode === 'customized'
                ? 'Identity with customized background'
                : 'Side-by-side comparison'}
            </p>
          </div>
        )}

        {(status === 'applying' || status === 'error') && (
          <div className="flex flex-col items-center gap-4 text-center max-w-md">
            {statusInfo.icon}
            <div>
              <h3 className="text-lg font-semibold">{statusInfo.title}</h3>
              <p className="text-sm text-muted-foreground mt-1">{statusInfo.description}</p>
            </div>
          </div>
        )}
      </div>

      <Separator className="my-6" />

      {/* Actions Bar */}
      <div className="space-y-4">
        {status === 'applying' && (
          <>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="font-medium">Processing</span>
                <span className="text-muted-foreground">{Math.round(progress * 100)}%</span>
              </div>
              <Progress value={progress * 100} className="h-2" />
            </div>

            {message && (
              <Alert>
                <Loader2 className="h-4 w-4 animate-spin" />
                <AlertDescription className="text-sm">{message}</AlertDescription>
              </Alert>
            )}
          </>
        )}

        {status === 'complete' && customizedResult && (
          <div className="space-y-3">
            <Alert>
              <CheckCircle2 className="h-4 w-4" />
              <AlertTitle>Success!</AlertTitle>
              <AlertDescription className="text-sm">
                Background has been applied successfully.
              </AlertDescription>
            </Alert>

            <div className="flex gap-2">
              <Button variant="outline" className="flex-1">
                <Download className="h-4 w-4 mr-2" />
                Download
              </Button>
              <Button className="flex-1">
                Proceed to Audit
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </div>
          </div>
        )}

        {status === 'error' && (
          <Alert variant="destructive">
            <XCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription className="text-sm">{error}</AlertDescription>
          </Alert>
        )}

        {status === 'idle' && !customizedResult && (
          <div className="text-center text-sm text-muted-foreground">
            Configure your background settings on the left, then click Apply to preview
          </div>
        )}
      </div>
    </div>
  )
}
