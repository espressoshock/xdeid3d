import { useState } from 'react'
import { useCustomization } from '@/hooks/useCustomization'
import { api } from '@/lib/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'
import { Box, RefreshCw, Info, Loader2 } from 'lucide-react'

const VOXEL_RESOLUTIONS = [32, 64, 128, 256, 512, 1024]

const QUALITY_DESCRIPTIONS: Record<number, { label: string; time: string; description: string }> = {
  32: { label: 'Very Low', time: '5-10s', description: 'Fast preview, low detail' },
  64: { label: 'Low', time: '10-15s', description: 'Quick preview, basic detail' },
  128: { label: 'Medium', time: '20-40s', description: 'Balanced quality and speed' },
  256: { label: 'High', time: '1-2min', description: 'Good detail, slower generation' },
  512: { label: 'Very High', time: '3-5min', description: 'High detail, slow generation' },
  1024: { label: 'Ultra', time: '8-12min', description: 'Maximum detail, very slow' },
}

export function MeshSettings() {
  const {
    baseIdentity,
    sessionId,
    meshGeneration,
    setVoxelResolution,
    startMeshRegeneration,
    setMeshGenerationStatus,
    setMeshUrl,
    setMeshError,
  } = useCustomization()

  const currentResolution = meshGeneration.voxelResolution
  const qualityInfo = QUALITY_DESCRIPTIONS[currentResolution]

  // Map resolution to slider index
  const resolutionToIndex = (res: number) => VOXEL_RESOLUTIONS.indexOf(res)
  const indexToResolution = (idx: number) => VOXEL_RESOLUTIONS[idx]

  const currentIndex = resolutionToIndex(currentResolution)

  const handleResolutionChange = (value: number[]) => {
    const newResolution = indexToResolution(value[0])
    setVoxelResolution(newResolution)
  }

  const handleRegenerateMesh = async () => {
    if (!baseIdentity || !sessionId) {
      console.error('[MeshSettings] Missing baseIdentity or sessionId')
      return
    }

    try {
      // Start regeneration (preserves old mesh URL)
      startMeshRegeneration()

      console.log('[MeshSettings] Regenerating mesh with voxel_res:', currentResolution)

      // Call mesh generation API with force_regenerate flag
      const response = await api.generateMesh({
        session_id: sessionId,
        latent_code_path: baseIdentity.latentCodePath,
        generator_path: baseIdentity.generatorPath,
        seed: baseIdentity.metadata?.seed,
        truncation: baseIdentity.metadata?.params?.truncation || baseIdentity.metadata?.truncation || 0.65,
        voxel_res: currentResolution,
        force_regenerate: true, // Force regeneration even if cached mesh exists
      })

      if (response.success && response.mesh_url) {
        console.log('[MeshSettings] Mesh regenerated successfully:', response.mesh_url)
        setMeshUrl(response.mesh_url)
      } else {
        throw new Error(response.error || response.message || 'Mesh regeneration failed')
      }
    } catch (error) {
      console.error('[MeshSettings] Mesh regeneration failed:', error)
      setMeshError(error instanceof Error ? error.message : 'Failed to regenerate mesh')
    }
  }

  // Check if quality setting changed since last generation
  // Parse voxel resolution from mesh URL (e.g., /api/media/session/mesh/seed0046_voxel128.ply)
  const getCurrentMeshResolution = () => {
    if (!meshGeneration.meshUrl) return null
    const match = meshGeneration.meshUrl.match(/voxel(\d+)\.ply/)
    return match ? parseInt(match[1]) : null
  }

  const currentMeshResolution = getCurrentMeshResolution()
  const hasQualityChanged =
    meshGeneration.meshUrl &&
    meshGeneration.status === 'complete' &&
    currentMeshResolution !== null &&
    currentMeshResolution !== currentResolution

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Box className="h-4 w-4" />
          Mesh Quality
        </CardTitle>
        <CardDescription>
          Adjust 3D mesh resolution for quality vs. speed trade-off
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-3">
          <div className="flex justify-between items-start">
            <div>
              <Label className="text-sm">Voxel Resolution</Label>
              <p className="text-xs text-muted-foreground mt-0.5">
                {qualityInfo.label} Quality
              </p>
            </div>
            <div className="text-right">
              <span className="text-sm font-mono font-medium">{currentResolution}</span>
              <p className="text-xs text-muted-foreground mt-0.5">
                ~{qualityInfo.time}
              </p>
            </div>
          </div>

          <Slider
            value={[currentIndex]}
            onValueChange={handleResolutionChange}
            min={0}
            max={VOXEL_RESOLUTIONS.length - 1}
            step={1}
            disabled={meshGeneration.isRegenerating}
            className="py-2"
          />

          <div className="flex justify-between text-xs text-muted-foreground px-0.5">
            <span>32</span>
            <span>64</span>
            <span>128</span>
            <span>256</span>
            <span>512</span>
            <span>1024</span>
          </div>

          <p className="text-xs text-muted-foreground">
            {qualityInfo.description}
          </p>
        </div>

        {/* Regeneration Progress */}
        {meshGeneration.isRegenerating && (
          <div className="space-y-2 p-3 bg-muted/50 rounded-md border border-border">
            <div className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin text-primary" />
              <div className="flex-1">
                <p className="text-sm font-medium">Regenerating Mesh</p>
                <p className="text-xs text-muted-foreground">
                  {meshGeneration.message || 'Processing...'}
                </p>
              </div>
              <span className="text-xs font-mono text-muted-foreground">
                {Math.round(Math.min(100, meshGeneration.progress * 100))}%
              </span>
            </div>
            <Progress value={Math.min(100, meshGeneration.progress * 100)} className="h-1.5" />
            <p className="text-xs text-muted-foreground">
              Old mesh remains visible until regeneration completes
            </p>
          </div>
        )}

        {/* Regeneration Error */}
        {meshGeneration.status === 'error' && meshGeneration.error && meshGeneration.meshUrl && (
          <Alert variant="destructive">
            <Info className="h-4 w-4" />
            <AlertDescription className="text-xs">
              <strong>Regeneration failed:</strong> {meshGeneration.error}
            </AlertDescription>
          </Alert>
        )}

        {/* Quality Changed Alert */}
        {hasQualityChanged && !meshGeneration.isRegenerating && meshGeneration.status !== 'error' && (
          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription className="text-xs">
              Quality setting changed. Click "Regenerate Mesh" to apply the new resolution.
            </AlertDescription>
          </Alert>
        )}

        <Button
          onClick={handleRegenerateMesh}
          variant="outline"
          className="w-full"
          disabled={meshGeneration.isRegenerating || !baseIdentity}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${meshGeneration.isRegenerating ? 'animate-spin' : ''}`} />
          {meshGeneration.isRegenerating ? 'Regenerating...' : 'Regenerate Mesh'}
        </Button>
      </CardContent>
    </Card>
  )
}
