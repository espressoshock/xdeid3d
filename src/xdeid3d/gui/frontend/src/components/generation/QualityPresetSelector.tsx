import { Label } from '@/components/ui/label'
import { QualityPreset, QUALITY_PRESETS } from '@/types/generation'
import { Check } from 'lucide-react'

interface QualityPresetSelectorProps {
  selected: QualityPreset
  onSelect: (preset: QualityPreset) => void
  disabled?: boolean
}

export function QualityPresetSelector({ selected, onSelect, disabled }: QualityPresetSelectorProps) {
  return (
    <div className="space-y-2">
      <Label>Quality Preset</Label>
      <div className="grid grid-cols-3 gap-2">
        {QUALITY_PRESETS.map((preset) => (
          <button
            key={preset.name}
            onClick={() => onSelect(preset)}
            disabled={disabled}
            className={`
              relative p-3 rounded-lg border-2 text-left transition-all
              ${
                selected.name === preset.name
                  ? 'border-primary bg-primary/5'
                  : 'border-muted hover:border-primary/50'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            {selected.name === preset.name && (
              <div className="absolute top-2 right-2 w-5 h-5 rounded-full bg-primary flex items-center justify-center">
                <Check className="h-3 w-3 text-primary-foreground" />
              </div>
            )}
            <div className="font-medium text-sm mb-1">{preset.label}</div>
            <div className="text-xs text-muted-foreground line-clamp-2">
              {preset.description}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
