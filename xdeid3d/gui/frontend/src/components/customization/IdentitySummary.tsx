import { useCustomization } from '@/hooks/useCustomization'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Dices, Upload, Pencil } from 'lucide-react'

export function IdentitySummary() {
  const { baseIdentity } = useCustomization()

  if (!baseIdentity) return null

  const { metadata } = baseIdentity
  const methodIcons = {
    seed: <Dices className="h-3.5 w-3.5" />,
    upload: <Upload className="h-3.5 w-3.5" />,
    text: <Pencil className="h-3.5 w-3.5" />,
  }

  const methodLabels = {
    seed: 'Seed Generation',
    upload: 'Image Upload (PTI)',
    text: 'Text Prompt',
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          Base Identity
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        {/* Generation Method */}
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">Method:</span>
          <Badge variant="secondary" className="gap-1.5">
            {methodIcons[metadata.method]}
            {methodLabels[metadata.method]}
          </Badge>
        </div>

        {/* Seed (if applicable) */}
        {metadata.seed !== undefined && (
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Seed:</span>
            <code className="text-xs bg-muted px-2 py-1 rounded">{metadata.seed}</code>
          </div>
        )}

        {/* Parameters */}
        {metadata.params && (
          <div className="space-y-1.5">
            <span className="text-muted-foreground text-xs">Parameters:</span>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {metadata.params.truncation !== undefined && (
                <div className="bg-muted/50 px-2 py-1 rounded">
                  <span className="text-muted-foreground">Trunc:</span>{' '}
                  <span className="font-mono">{metadata.params.truncation}</span>
                </div>
              )}
              {metadata.params.nrr !== undefined && (
                <div className="bg-muted/50 px-2 py-1 rounded">
                  <span className="text-muted-foreground">NRR:</span>{' '}
                  <span className="font-mono">{metadata.params.nrr}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Preview Thumbnail */}
        <div className="mt-3">
          <img
            src={baseIdentity.imageUrl}
            alt="Base identity"
            className="w-full rounded border"
          />
        </div>
      </CardContent>
    </Card>
  )
}
