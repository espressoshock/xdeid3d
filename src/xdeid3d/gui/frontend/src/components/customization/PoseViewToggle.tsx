import { useCustomization } from '@/hooks/useCustomization'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Box, Video, Loader2 } from 'lucide-react'

export function PoseViewToggle() {
  const { posePreview, setPoseViewMode } = useCustomization()

  const { viewMode, status } = posePreview

  const isRendering = status === 'rendering'

  return (
    <div className="flex justify-center py-3">
      <ToggleGroup
        type="single"
        value={viewMode}
        onValueChange={(value) => {
          if (value) setPoseViewMode(value as '3d' | 'rendered')
        }}
        className="bg-muted p-1 rounded-lg"
      >
        <ToggleGroupItem
          value="3d"
          className="data-[state=on]:bg-background data-[state=on]:shadow-sm px-4 relative"
        >
          <Box className="h-4 w-4 mr-2" />
          3D Model
        </ToggleGroupItem>

        <ToggleGroupItem
          value="rendered"
          className="data-[state=on]:bg-background data-[state=on]:shadow-sm px-4 relative"
          disabled={isRendering && viewMode === '3d'}
        >
          {isRendering && viewMode === 'rendered' ? (
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <Video className="h-4 w-4 mr-2" />
          )}
          Rendered Pose
          {isRendering && viewMode === 'rendered' && (
            <span className="ml-2 text-xs text-muted-foreground">
              ({Math.round(posePreview.progress * 100)}%)
            </span>
          )}
        </ToggleGroupItem>
      </ToggleGroup>
    </div>
  )
}
