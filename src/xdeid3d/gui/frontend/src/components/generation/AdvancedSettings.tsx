import { Separator } from '@/components/ui/separator'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { ChevronDown, ChevronUp, Settings2 } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface AdvancedSettingsProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  params: {
    truncation: number
    nrr: number
    sampleMult: number
  }
  onParamsChange: (params: any) => void
  disabled?: boolean
}

export function AdvancedSettings({
  open,
  onOpenChange,
  params,
  onParamsChange,
  disabled,
}: AdvancedSettingsProps) {
  const updateParam = (key: string, value: number) => {
    onParamsChange({ ...params, [key]: value })
  }

  return (
    <div className="space-y-3">
      <Separator />

      <Button
        variant="ghost"
        className="w-full justify-between h-auto py-2"
        onClick={() => onOpenChange(!open)}
        disabled={disabled}
      >
        <div className="flex items-center gap-2">
          <Settings2 className="h-4 w-4" />
          <span className="font-medium">Advanced Settings</span>
        </div>
        {open ? (
          <ChevronUp className="h-4 w-4" />
        ) : (
          <ChevronDown className="h-4 w-4" />
        )}
      </Button>

      {open && (
        <div className="space-y-4 pt-2">
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label className="text-sm">Truncation</Label>
              <span className="text-sm text-muted-foreground">{params.truncation.toFixed(2)}</span>
            </div>
            <Slider
              value={[params.truncation]}
              onValueChange={(v) => updateParam('truncation', v[0])}
              min={0.5}
              max={0.8}
              step={0.05}
              disabled={disabled}
            />
            <p className="text-xs text-muted-foreground">
              Lower = higher quality, less diversity
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <Label className="text-sm">Neural Rendering Resolution</Label>
              <span className="text-sm text-muted-foreground">{params.nrr}</span>
            </div>
            <Slider
              value={[params.nrr]}
              onValueChange={(v) => updateParam('nrr', v[0])}
              min={64}
              max={1024}
              step={64}
              disabled={disabled}
            />
            <p className="text-xs text-muted-foreground">
              Higher = sharper, but slower (64, 128, 256, 512, 768, 1024)
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <Label className="text-sm">Sample Multiplier</Label>
              <span className="text-sm text-muted-foreground">{params.sampleMult.toFixed(1)}</span>
            </div>
            <Slider
              value={[params.sampleMult]}
              onValueChange={(v) => updateParam('sampleMult', v[0])}
              min={1.0}
              max={4.0}
              step={0.5}
              disabled={disabled}
            />
            <p className="text-xs text-muted-foreground">
              Higher = fewer artifacts, but slower (1.0 - 4.0)
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
