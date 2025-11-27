import { useState } from 'react'
import { useCustomization } from '@/hooks/useCustomization'
import { Card } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Separator } from '@/components/ui/separator'
import { ChevronDown, ChevronRight, Settings2 } from 'lucide-react'

export function PosePreviewSettings() {
  const { posePreview, setPosePreviewSettings } = useCustomization()
  const [isExpanded, setIsExpanded] = useState(false)

  const { settings } = posePreview

  // Calculate duration from frames and fps
  const duration = (settings.frames / settings.fps).toFixed(2)

  return (
    <Card className="p-4">
      {/* Collapsible Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between text-sm font-medium hover:text-primary transition-colors"
      >
        <div className="flex items-center gap-2">
          <Settings2 className="h-4 w-4" />
          <span>Advanced Settings</span>
        </div>
        {isExpanded ? (
          <ChevronDown className="h-4 w-4" />
        ) : (
          <ChevronRight className="h-4 w-4" />
        )}
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <>
          <Separator className="my-3" />

          <div className="space-y-4">
            {/* Preview Frames Slider */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <Label className="text-xs">Preview Frames</Label>
                <span className="text-xs font-mono text-muted-foreground">{settings.frames}</span>
              </div>
              <Slider
                value={[settings.frames]}
                onValueChange={(value) =>
                  setPosePreviewSettings({ frames: value[0] })
                }
                min={30}
                max={240}
                step={30}
                className="w-full"
              />
              <div className="relative h-4 text-xs text-muted-foreground">
                <span className="absolute left-0">30</span>
                <span className="absolute left-[14.286%] -translate-x-1/2">60</span>
                <span className="absolute left-[42.857%] -translate-x-1/2">120</span>
                <span className="absolute right-0">240</span>
              </div>
            </div>

            {/* FPS Selector */}
            <div className="space-y-2">
              <Label className="text-xs">Frames Per Second</Label>
              <Select
                value={settings.fps.toString()}
                onValueChange={(value) =>
                  setPosePreviewSettings({ fps: parseInt(value) })
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select FPS" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="24">24 fps (cinematic)</SelectItem>
                  <SelectItem value="30">30 fps (standard)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Quality Radio Group */}
            <div className="space-y-2">
              <Label className="text-xs">Render Quality</Label>
              <RadioGroup
                value={settings.quality}
                onValueChange={(value) =>
                  setPosePreviewSettings({ quality: value as 'low' | 'medium' | 'high' | 'best' })
                }
                className="space-y-2"
              >
                <div className="flex items-start space-x-2">
                  <RadioGroupItem value="low" id="quality-low" />
                  <div className="flex-1">
                    <label
                      htmlFor="quality-low"
                      className="text-xs font-medium cursor-pointer"
                    >
                      Low
                    </label>
                    <p className="text-xs text-muted-foreground">
                      Fast preview (~30s render, nrr=64)
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-2">
                  <RadioGroupItem value="medium" id="quality-medium" />
                  <div className="flex-1">
                    <label
                      htmlFor="quality-medium"
                      className="text-xs font-medium cursor-pointer"
                    >
                      Medium
                    </label>
                    <p className="text-xs text-muted-foreground">
                      Balanced quality (~1-2min render, nrr=128)
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-2">
                  <RadioGroupItem value="high" id="quality-high" />
                  <div className="flex-1">
                    <label
                      htmlFor="quality-high"
                      className="text-xs font-medium cursor-pointer"
                    >
                      High
                    </label>
                    <p className="text-xs text-muted-foreground">
                      Best quality (~3-5min render, nrr=256)
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-2">
                  <RadioGroupItem value="best" id="quality-best" />
                  <div className="flex-1">
                    <label
                      htmlFor="quality-best"
                      className="text-xs font-medium cursor-pointer"
                    >
                      Best
                    </label>
                    <p className="text-xs text-muted-foreground">
                      Maximum quality (~10-15min render, nrr=512)
                    </p>
                  </div>
                </div>
              </RadioGroup>
            </div>

            {/* Info Display */}
            <div className="pt-2 border-t">
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-muted-foreground">Duration:</span>{' '}
                  <span className="font-mono">{duration}s</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Total Frames:</span>{' '}
                  <span className="font-mono">{settings.frames}</span>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </Card>
  )
}
