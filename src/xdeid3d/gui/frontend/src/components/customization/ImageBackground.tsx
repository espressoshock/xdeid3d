import { useCallback, useState } from 'react'
import { useCustomization } from '@/hooks/useCustomization'
import { Card } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Upload, X, ImageIcon } from 'lucide-react'

export function ImageBackground() {
  const { backgroundSettings, updateBackgroundSettings } = useCustomization()
  const [preview, setPreview] = useState<string | null>(backgroundSettings.imageUrl || null)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (PNG, JPG, etc.)')
      return
    }

    // Validate file size (10MB max)
    const maxSize = 10 * 1024 * 1024
    if (file.size > maxSize) {
      setError('Image must be smaller than 10MB')
      return
    }

    setError(null)

    // Create preview
    const reader = new FileReader()
    reader.onload = (event) => {
      const url = event.target?.result as string
      setPreview(url)
      updateBackgroundSettings({
        imageFile: file,
        imageUrl: url,
      })
    }
    reader.readAsDataURL(file)
  }, [updateBackgroundSettings])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) {
      const input = document.createElement('input')
      input.type = 'file'
      const dataTransfer = new DataTransfer()
      dataTransfer.items.add(file)
      input.files = dataTransfer.files
      handleFileChange({ target: input } as any)
    }
  }, [handleFileChange])

  const clearImage = () => {
    setPreview(null)
    setError(null)
    updateBackgroundSettings({
      imageFile: undefined,
      imageUrl: undefined,
    })
  }

  const imageFit = backgroundSettings.imageFit || 'cover'

  return (
    <Card className="p-4 space-y-4">
      <div className="space-y-2">
        <Label>Upload Background Image</Label>
        {!preview ? (
          <div
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            className="border-2 border-dashed rounded-lg p-8 text-center hover:border-primary/50 transition-colors cursor-pointer"
          >
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
              id="background-upload"
            />
            <label
              htmlFor="background-upload"
              className="cursor-pointer flex flex-col items-center gap-2"
            >
              <Upload className="h-8 w-8 text-muted-foreground" />
              <div>
                <p className="text-sm font-medium">Click to upload or drag and drop</p>
                <p className="text-xs text-muted-foreground mt-1">
                  PNG, JPG up to 10MB
                </p>
              </div>
            </label>
          </div>
        ) : (
          <div className="relative">
            <div className="border rounded-lg overflow-hidden">
              <img
                src={preview}
                alt="Background preview"
                className="w-full h-48 object-cover"
              />
            </div>
            <Button
              variant="destructive"
              size="icon"
              className="absolute top-2 right-2"
              onClick={clearImage}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        )}

        {error && (
          <Alert variant="destructive" className="mt-2">
            <AlertDescription className="text-xs">{error}</AlertDescription>
          </Alert>
        )}
      </div>

      {preview && (
        <div className="space-y-2">
          <Label>Image Fit</Label>
          <RadioGroup
            value={imageFit}
            onValueChange={(value) =>
              updateBackgroundSettings({ imageFit: value as typeof imageFit })
            }
          >
            <div className="grid gap-3">
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="cover" id="cover" />
                <Label htmlFor="cover" className="font-normal cursor-pointer flex-1">
                  <div>
                    <p className="text-sm">Cover</p>
                    <p className="text-xs text-muted-foreground">
                      Fill the entire background, may crop image
                    </p>
                  </div>
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="contain" id="contain" />
                <Label htmlFor="contain" className="font-normal cursor-pointer flex-1">
                  <div>
                    <p className="text-sm">Contain</p>
                    <p className="text-xs text-muted-foreground">
                      Fit entire image, may show letterboxing
                    </p>
                  </div>
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="stretch" id="stretch" />
                <Label htmlFor="stretch" className="font-normal cursor-pointer flex-1">
                  <div>
                    <p className="text-sm">Stretch</p>
                    <p className="text-xs text-muted-foreground">
                      Stretch to fill, may distort image
                    </p>
                  </div>
                </Label>
              </div>
            </div>
          </RadioGroup>
        </div>
      )}

      {!preview && (
        <div className="rounded-lg border p-4 bg-muted/50 flex items-center justify-center h-32">
          <div className="text-center text-muted-foreground">
            <ImageIcon className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-xs">No background image uploaded</p>
          </div>
        </div>
      )}
    </Card>
  )
}
