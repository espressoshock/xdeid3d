import { useState, useRef, DragEvent } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Separator } from '@/components/ui/separator'
import { Switch } from '@/components/ui/switch'
import { useGeneration } from '@/hooks/useGeneration'
import { QualityPresetSelector } from './QualityPresetSelector'
import { api } from '@/lib/api'
import { QUALITY_PRESETS, QualityPreset } from '@/types/generation'
import { Upload, X, AlertCircle, Image as ImageIcon, Sparkles, Loader2, Info, Settings2, ChevronDown, ChevronUp } from 'lucide-react'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

interface ImageUploaderProps {
  sessionId: string
  isConnected: boolean
  sendMessage: (message: any) => void
}

export function ImageUploader({ sessionId, isConnected, sendMessage }: ImageUploaderProps) {
  const { status } = useGeneration()
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // PTI-specific settings
  const [selectedPreset, setSelectedPreset] = useState<QualityPreset>(QUALITY_PRESETS[1]) // Balanced
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [wSteps, setWSteps] = useState(500)
  const [ptiSteps, setPtiSteps] = useState(350)

  // Noise optimization settings
  const [optimizeNoise, setOptimizeNoise] = useState(false)
  const [initialNoiseFactor, setInitialNoiseFactor] = useState(0.05)
  const [noiseRampLength, setNoiseRampLength] = useState(0.75)
  const [regularizeNoiseWeight, setRegularizeNoiseWeight] = useState(100000) // 1e5

  const isGenerating = status === 'generating'
  const canGenerate = isConnected && !isGenerating && !isUploading && selectedFile !== null

  const handleDrag = (e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (file: File) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file')
      return
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB')
      return
    }

    setSelectedFile(file)

    // Create preview
    const reader = new FileReader()
    reader.onloadend = () => {
      setPreviewUrl(reader.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleClearFile = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setUploadError(null)
    setUploadProgress(0)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handlePresetChange = (preset: QualityPreset) => {
    setSelectedPreset(preset)
    // Always update PTI steps when preset changes
    if (preset.ptiSteps) {
      setWSteps(preset.ptiSteps.wSteps)
      setPtiSteps(preset.ptiSteps.ptiSteps)
    }
  }

  const handleGenerate = async () => {
    if (!selectedFile) return

    setUploadError(null)
    setIsUploading(true)
    setUploadProgress(0)

    try {
      console.log('[ImageUploader] Uploading image:', {
        sessionId,
        fileName: selectedFile.name,
        fileSize: selectedFile.size,
        fileType: selectedFile.type,
      })

      // Step 1: Upload file
      const uploadResponse = await api.uploadImage(selectedFile)

      console.log('[ImageUploader] Upload response:', uploadResponse)

      if (!uploadResponse.success) {
        throw new Error(uploadResponse.error || 'Upload failed')
      }

      if (!uploadResponse.data || !uploadResponse.data.upload_id) {
        console.error('[ImageUploader] Invalid response structure:', uploadResponse)
        throw new Error('Invalid upload response: missing upload_id')
      }

      const uploadId = uploadResponse.data.upload_id
      setUploadProgress(100)
      setIsUploading(false)

      console.log('[ImageUploader] Upload successful, starting PTI projection:', {
        uploadId,
        wSteps,
        ptiSteps,
        optimizeNoise,
      })

      // Step 2: Start PTI via WebSocket
      sendMessage({
        type: 'start_pti',
        upload_id: uploadId,
        session_id: sessionId,
        params: {
          ...selectedPreset.params,
          w_steps: wSteps,
          pti_steps: ptiSteps,
          generate_video: false, // Always generate video for PTI
          optimize_noise: optimizeNoise,
          initial_noise_factor: initialNoiseFactor,
          noise_ramp_length: noiseRampLength,
          regularize_noise_weight: regularizeNoiseWeight,
        },
      })

    } catch (error) {
      console.error('[ImageUploader] Upload error:', error)
      setUploadError(error instanceof Error ? error.message : 'Upload failed')
      setIsUploading(false)
    }
  }

  return (
    <div className="space-y-4">
      {/* Image Upload Card */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Image Upload (PTI)</CardTitle>
          <CardDescription>
            Upload a face image to reconstruct as a 3D head
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!selectedFile ? (
            <div
              className={`
                border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
                transition-colors hover:border-primary/50
                ${dragActive ? 'border-primary bg-primary/5' : 'border-muted'}
                ${isGenerating ? 'opacity-50 cursor-not-allowed' : ''}
              `}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => !isGenerating && fileInputRef.current?.click()}
            >
              <div className="flex flex-col items-center gap-2">
                <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                  <Upload className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <p className="font-medium">Drop image here or click to browse</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    PNG, JPG up to 10MB
                  </p>
                </div>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                accept="image/png,image/jpeg,image/jpg"
                onChange={handleChange}
                disabled={isGenerating}
              />
            </div>
          ) : (
            <div className="space-y-3">
              <div className="relative rounded-lg overflow-hidden border">
                {previewUrl && (
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="w-full h-64 object-cover"
                  />
                )}
                <Button
                  variant="destructive"
                  size="icon"
                  className="absolute top-2 right-2"
                  onClick={handleClearFile}
                  disabled={isGenerating || isUploading}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <div className="flex items-start gap-2 text-sm">
                <ImageIcon className="h-4 w-4 mt-0.5 text-muted-foreground" />
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{selectedFile.name}</p>
                  <p className="text-muted-foreground">
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
            </div>
          )}

          {uploadError && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="text-xs">{uploadError}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Settings Card */}
      {selectedFile && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">PTI Settings</CardTitle>
            <CardDescription>
              Configure quality and projection parameters
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <QualityPresetSelector
              selected={selectedPreset}
              onSelect={handlePresetChange}
              disabled={isGenerating || isUploading}
            />

            <Separator />

            {/* Advanced PTI Settings */}
            <div className="space-y-3">
              <Button
                variant="ghost"
                className="w-full justify-between h-auto py-2"
                onClick={() => setShowAdvanced(!showAdvanced)}
                disabled={isGenerating || isUploading}
              >
                <div className="flex items-center gap-2">
                  <Settings2 className="h-4 w-4" />
                  <span className="font-medium">Advanced PTI Settings</span>
                </div>
                {showAdvanced ? (
                  <ChevronUp className="h-4 w-4" />
                ) : (
                  <ChevronDown className="h-4 w-4" />
                )}
              </Button>

              {showAdvanced && (
                <div className="space-y-4 pt-2">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label className="text-sm">W-Projection Steps</Label>
                      <span className="text-sm text-muted-foreground">{wSteps}</span>
                    </div>
                    <Slider
                      value={[wSteps]}
                      onValueChange={(v) => setWSteps(v[0])}
                      min={100}
                      max={3000}
                      step={100}
                      disabled={isGenerating || isUploading}
                    />
                    <p className="text-xs text-muted-foreground">
                      More steps = better quality, longer time (~{Math.round(wSteps / 100)} min)
                    </p>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label className="text-sm">PTI Fine-tuning Steps</Label>
                      <span className="text-sm text-muted-foreground">{ptiSteps}</span>
                    </div>
                    <Slider
                      value={[ptiSteps]}
                      onValueChange={(v) => setPtiSteps(v[0])}
                      min={100}
                      max={2000}
                      step={100}
                      disabled={isGenerating || isUploading}
                    />
                    <p className="text-xs text-muted-foreground">
                      More steps = better reconstruction (~{Math.round(ptiSteps / 100)} min)
                    </p>
                  </div>
                </div>
              )}
            </div>

            <Separator />

            {/* Noise Optimization Section */}
            <TooltipProvider>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="optimize-noise" className="text-sm font-medium">
                      Noise Optimization
                    </Label>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p className="text-xs">
                          Enables optimization of noise inputs for finer details and textures.
                          This can improve quality but increases projection time by ~20-30%.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <Switch
                    id="optimize-noise"
                    checked={optimizeNoise}
                    onCheckedChange={setOptimizeNoise}
                    disabled={isGenerating || isUploading}
                  />
                </div>

                {optimizeNoise && (
                  <div className="space-y-4 pl-4 border-l-2 border-primary/20">
                    {/* Initial Noise Factor */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Label className="text-sm">Initial Noise Factor</Label>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Info className="h-3 w-3 text-muted-foreground cursor-help" />
                            </TooltipTrigger>
                            <TooltipContent className="max-w-xs">
                              <p className="text-xs">
                                Starting strength of noise injection. Higher values add more variation
                                but may reduce stability. Default: 0.05
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <span className="text-sm text-muted-foreground">{initialNoiseFactor.toFixed(3)}</span>
                      </div>
                      <Slider
                        value={[initialNoiseFactor * 1000]} // Scale for better UX
                        onValueChange={(v) => setInitialNoiseFactor(v[0] / 1000)}
                        min={10}
                        max={200}
                        step={5}
                        disabled={isGenerating || isUploading}
                      />
                      <p className="text-xs text-muted-foreground">
                        Range: 0.01 - 0.20 (lower = more stable, higher = more detail variation)
                      </p>
                    </div>

                    {/* Noise Ramp Length */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Label className="text-sm">Noise Ramp Length</Label>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Info className="h-3 w-3 text-muted-foreground cursor-help" />
                            </TooltipTrigger>
                            <TooltipContent className="max-w-xs">
                              <p className="text-xs">
                                Controls how long noise persists during optimization (as fraction of total steps).
                                Longer = more detail exploration. Default: 0.75 (75% of steps)
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <span className="text-sm text-muted-foreground">{(noiseRampLength * 100).toFixed(0)}%</span>
                      </div>
                      <Slider
                        value={[noiseRampLength * 100]}
                        onValueChange={(v) => setNoiseRampLength(v[0] / 100)}
                        min={25}
                        max={100}
                        step={5}
                        disabled={isGenerating || isUploading}
                      />
                      <p className="text-xs text-muted-foreground">
                        Percentage of optimization steps with noise active
                      </p>
                    </div>

                    {/* Noise Regularization Weight */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Label className="text-sm">Noise Regularization</Label>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Info className="h-3 w-3 text-muted-foreground cursor-help" />
                            </TooltipTrigger>
                            <TooltipContent className="max-w-xs">
                              <p className="text-xs">
                                Strength of noise smoothness constraint. Higher values = smoother noise patterns,
                                lower values = more detailed but potentially noisy. Default: 100k (1e5)
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <span className="text-sm text-muted-foreground">{(regularizeNoiseWeight / 1000).toFixed(0)}k</span>
                      </div>
                      <Slider
                        value={[regularizeNoiseWeight / 1000]} // Scale to k for UX
                        onValueChange={(v) => setRegularizeNoiseWeight(v[0] * 1000)}
                        min={10}
                        max={500}
                        step={10}
                        disabled={isGenerating || isUploading}
                      />
                      <p className="text-xs text-muted-foreground">
                        Range: 10k - 500k (lower = more detail, higher = smoother)
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </TooltipProvider>
          </CardContent>
        </Card>
      )}

      {/* Info Alert */}
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>PTI Process</AlertTitle>
        <AlertDescription className="text-xs">
          The image will go through two stages: (1) W-space projection and
          (2) Generator fine-tuning. Total time: ~{Math.round((wSteps + ptiSteps) / 100)} minutes.
        </AlertDescription>
      </Alert>

      {/* Tips Alert */}
      {selectedFile && (
        <Alert>
          <Sparkles className="h-4 w-4" />
          <AlertDescription className="text-xs">
            Tip: For best results, use a clear frontal face photo with good lighting and neutral expression.
            Higher quality presets and more steps produce better reconstruction but take longer.
          </AlertDescription>
        </Alert>
      )}

      {/* Generate Button */}
      <Button
        className="w-full"
        size="lg"
        onClick={handleGenerate}
        disabled={!canGenerate}
      >
        {isUploading ? (
          <>
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            Uploading... {uploadProgress}%
          </>
        ) : isGenerating ? (
          'Processing PTI...'
        ) : (
          'Start PTI Projection'
        )}
      </Button>
    </div>
  )
}
