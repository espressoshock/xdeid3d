import { useState } from 'react'
import { useCustomization } from '@/hooks/useCustomization'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { AlertCircle, Palette, Video } from 'lucide-react'
import { BackgroundSelector } from './BackgroundSelector'
import { CustomizationPreview } from './CustomizationPreview'
import { IdentitySummary } from './IdentitySummary'
import { CameraPosePanel } from './CameraPosePanel'

type CustomizationStage = 'background' | 'camera'

export function CustomizationPanel() {
  const { baseIdentity } = useCustomization()
  const [stage, setStage] = useState<CustomizationStage>('background')

  // If no base identity from Stage 1, show warning
  if (!baseIdentity) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <Alert className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>No Identity Generated</AlertTitle>
          <AlertDescription>
            Please complete the generation stage first to create a base identity before customizing.
          </AlertDescription>
        </Alert>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      {/* Sub-stage tabs */}
      <div className="border-b px-6 pt-4">
        <Tabs value={stage} onValueChange={(v) => setStage(v as CustomizationStage)}>
          <TabsList className="grid w-full max-w-md grid-cols-2">
            <TabsTrigger value="background" className="gap-2">
              <Palette className="h-3.5 w-3.5" />
              Background
            </TabsTrigger>
            <TabsTrigger value="camera" className="gap-2">
              <Video className="h-3.5 w-3.5" />
              Camera Pose
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {/* Content based on selected stage */}
      <div className="flex-1 overflow-hidden">
        {stage === 'background' ? (
          <div className="h-full flex">
            {/* Left Panel - Customization Controls */}
            <div className="w-[420px] border-r flex flex-col">
              <div className="p-6 flex-1 overflow-y-auto">
                <div className="mb-4">
                  <h2 className="text-lg font-semibold mb-1">Customize Background</h2>
                  <p className="text-sm text-muted-foreground">
                    Alter the background of your generated identity
                  </p>
                </div>

                <Separator className="my-4" />

                {/* Identity Summary */}
                <IdentitySummary />

                <Separator className="my-6" />

                {/* Background Customization */}
                <BackgroundSelector />
              </div>

              {/* Connection Status */}
              <div className="px-6 py-3 border-t">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <div className="h-2 w-2 rounded-full bg-green-500" />
                  Ready to customize
                </div>
              </div>
            </div>

            {/* Right Panel - Preview */}
            <div className="flex-1 overflow-hidden">
              <CustomizationPreview />
            </div>
          </div>
        ) : (
          <CameraPosePanel />
        )}
      </div>
    </div>
  )
}
