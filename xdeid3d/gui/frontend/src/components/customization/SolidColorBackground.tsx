import { useState } from 'react'
import { useCustomization } from '@/hooks/useCustomization'
import { Card } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { HexColorPicker } from 'react-colorful'
import { Check } from 'lucide-react'

const COLOR_PRESETS = [
  { name: 'Black', value: '#000000' },
  { name: 'Dark Gray', value: '#1a1a1a' },
  { name: 'White', value: '#ffffff' },
  { name: 'Light Gray', value: '#e5e5e5' },
  { name: 'Studio Blue', value: '#1e3a5f' },
  { name: 'Studio Green', value: '#0f3d2e' },
  { name: 'Deep Purple', value: '#2d1b3d' },
  { name: 'Warm Brown', value: '#3d2817' },
]

export function SolidColorBackground() {
  const { backgroundSettings, updateBackgroundSettings } = useCustomization()
  const [open, setOpen] = useState(false)

  const currentColor = backgroundSettings.solidColor || '#1a1a1a'

  const handleColorChange = (color: string) => {
    updateBackgroundSettings({ solidColor: color })
  }

  return (
    <Card className="p-4 space-y-4">
      <div className="space-y-2">
        <Label>Color Presets</Label>
        <div className="grid grid-cols-4 gap-2">
          {COLOR_PRESETS.map((preset) => (
            <button
              key={preset.value}
              onClick={() => handleColorChange(preset.value)}
              className="relative group"
              title={preset.name}
            >
              <div
                className="h-12 w-full rounded border-2 transition-all hover:scale-105"
                style={{
                  backgroundColor: preset.value,
                  borderColor: currentColor === preset.value ? 'hsl(var(--primary))' : 'hsl(var(--border))',
                }}
              >
                {currentColor === preset.value && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Check className="h-5 w-5 text-primary-foreground drop-shadow-md" />
                  </div>
                )}
              </div>
              <p className="text-xs text-muted-foreground mt-1 truncate">{preset.name}</p>
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-2">
        <Label>Custom Color</Label>
        <div className="flex gap-2">
          <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
              <Button variant="outline" className="w-full justify-start gap-2">
                <div
                  className="h-4 w-4 rounded border"
                  style={{ backgroundColor: currentColor }}
                />
                <span className="flex-1 text-left font-mono text-sm">{currentColor}</span>
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-3" align="start">
              <HexColorPicker color={currentColor} onChange={handleColorChange} />
            </PopoverContent>
          </Popover>
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="hex-input">Hex Value</Label>
        <Input
          id="hex-input"
          value={currentColor}
          onChange={(e) => handleColorChange(e.target.value)}
          placeholder="#000000"
          className="font-mono"
        />
      </div>

      <div className="pt-2">
        <div className="rounded-lg border p-4 flex items-center justify-center bg-muted/50">
          <div
            className="h-24 w-24 rounded-full border-4 border-background shadow-lg"
            style={{ backgroundColor: currentColor }}
          />
        </div>
        <p className="text-xs text-center text-muted-foreground mt-2">
          Preview of selected color
        </p>
      </div>
    </Card>
  )
}
