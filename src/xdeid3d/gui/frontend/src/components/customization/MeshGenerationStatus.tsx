import { Loader2, Box, CheckCircle2, XCircle } from 'lucide-react'
import { Progress } from '@/components/ui/progress'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

interface MeshGenerationStatusProps {
  status: 'idle' | 'generating' | 'complete' | 'error'
  progress: number
  message: string
  error?: string | null
  onRetry?: () => void
}

export function MeshGenerationStatus({
  status,
  progress,
  message,
  error,
  onRetry,
}: MeshGenerationStatusProps) {
  if (status === 'generating') {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <Card className="max-w-md w-full p-6 space-y-6">
          <div className="flex items-center gap-4">
            <div className="relative">
              <Loader2 className="h-12 w-12 animate-spin text-primary" />
              <Box className="h-6 w-6 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-primary" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold mb-1">Generating 3D Mesh</h3>
              <p className="text-sm text-muted-foreground">
                {message || 'Extracting 3D model from generated identity...'}
              </p>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Progress</span>
              <span>{Math.round(Math.min(100, progress * 100))}%</span>
            </div>
            <Progress value={Math.min(100, progress * 100)} className="h-2" />
          </div>

          <div className="space-y-1 text-xs text-muted-foreground">
            <p>• This process takes 20-40 seconds (voxel_res: 128)</p>
            <p>• The 3D mesh will be used for camera pose customization</p>
            <p>• Meshes are cached - subsequent loads are instant</p>
          </div>
        </Card>
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <Card className="max-w-md w-full p-6 space-y-4 border-destructive">
          <div className="flex items-center gap-3">
            <XCircle className="h-10 w-10 text-destructive flex-shrink-0" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-destructive mb-1">Mesh Generation Failed</h3>
              <p className="text-sm text-muted-foreground">
                {error || 'An error occurred while generating the 3D mesh'}
              </p>
            </div>
          </div>

          {onRetry && (
            <Button
              onClick={onRetry}
              variant="outline"
              className="w-full"
            >
              Try Again
            </Button>
          )}

          <div className="text-xs text-muted-foreground space-y-1">
            <p className="font-medium">Possible solutions:</p>
            <ul className="list-disc list-inside space-y-0.5 ml-2">
              <li>Check if the identity was generated correctly</li>
              <li>Verify all mesh dependencies are installed (plyfile, scikit-image, PyMCubes)</li>
              <li>Ensure the SphereHead model is loaded correctly</li>
            </ul>
          </div>
        </Card>
      </div>
    )
  }

  if (status === 'complete') {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <Card className="max-w-md w-full p-6 space-y-4 border-green-500/50">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="h-10 w-10 text-green-500 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-green-500 mb-1">Mesh Ready!</h3>
              <p className="text-sm text-muted-foreground">
                3D mesh has been generated successfully
              </p>
            </div>
          </div>

          <div className="text-xs text-muted-foreground">
            You can now interact with the 3D model in the viewer to customize camera poses.
          </div>
        </Card>
      </div>
    )
  }

  // Idle state (should not normally be shown)
  return null
}
